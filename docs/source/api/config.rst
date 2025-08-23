Configuration
=============

.. module:: openhcs.core.config

OpenHCS uses a hierarchical configuration system with immutable dataclasses. The main configuration object is ``GlobalPipelineConfig``, which contains specialized sub-configurations for different system components.

GlobalPipelineConfig
--------------------

.. autoclass:: GlobalPipelineConfig
   :members:
   :undoc-members:
   :show-inheritance:

The root configuration object for OpenHCS pipeline sessions. This object is intended to be instantiated at application startup and treated as immutable.

**Key Features**:

- **Immutable Design**: All configuration objects are frozen dataclasses
- **Hierarchical Structure**: Specialized sub-configurations for different components
- **Sensible Defaults**: Works out-of-the-box with minimal configuration
- **Environment Integration**: Reads from environment variables where appropriate

VFSConfig
---------

.. autoclass:: VFSConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration for Virtual File System (VFS) operations, controlling how intermediate and final results are stored.

PathPlanningConfig
------------------

.. autoclass:: PathPlanningConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration for pipeline path planning, defining directory suffixes and output locations.

ZarrConfig
----------

.. autoclass:: ZarrConfig
   :members:
   :undoc-members:
   :show-inheritance:

Configuration for ZARR storage backend, including compression and chunking strategies.

Usage Examples
--------------

Basic Configuration
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.config import GlobalPipelineConfig

    # Use default configuration
    config = GlobalPipelineConfig()

    # Or get the default instance
    from openhcs.core.config import get_default_global_config
    config = get_default_global_config()

Custom Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.core.config import (
        GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig,
        MaterializationBackend, ZarrCompressor, ZarrChunkStrategy
    )
    from openhcs.constants.constants import Backend, Microscope

    # Custom configuration for large-scale processing
    config = GlobalPipelineConfig(
        num_workers=8,
        path_planning=PathPlanningConfig(
            output_dir_suffix="_processed",
            global_output_folder="/data/hcs_results",
            materialization_results_path="analysis"
        ),
        vfs=VFSConfig(
            intermediate_backend=Backend.MEMORY,
            materialization_backend=MaterializationBackend.ZARR
        ),
        zarr=ZarrConfig(
            store_name="images.zarr",
            compressor=ZarrCompressor.ZSTD,
            compression_level=3,
            shuffle=True,
            chunk_strategy=ZarrChunkStrategy.SINGLE,
            ome_zarr_metadata=True,
            write_plate_metadata=True
        ),
        microscope=Microscope.IMAGEXPRESS
    )

Production Example
^^^^^^^^^^^^^^^^^^

Complete configuration from gold standard script:

.. code-block:: python

    from openhcs.core.config import (
        GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig,
        MaterializationBackend, ZarrCompressor, ZarrChunkStrategy
    )
    from openhcs.constants.constants import Backend, Microscope

    # Production configuration for neurite analysis
    production_config = GlobalPipelineConfig(
        num_workers=5,
        path_planning=PathPlanningConfig(
            output_dir_suffix="_stitched",
            global_output_folder="/home/ts/nvme_usb/OpenHCS/",
            materialization_results_path="results"
        ),
        vfs=VFSConfig(
            intermediate_backend=Backend.MEMORY,
            materialization_backend=MaterializationBackend.ZARR
        ),
        zarr=ZarrConfig(
            store_name="images.zarr",
            compressor=ZarrCompressor.ZSTD,
            compression_level=1,
            shuffle=True,
            chunk_strategy=ZarrChunkStrategy.SINGLE,
            ome_zarr_metadata=True,
            write_plate_metadata=True
        ),
        microscope=Microscope.AUTO,
        use_threading=False  # Use multiprocessing for better performance
    )

    # Use with orchestrator
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(
        plate_path="/path/to/plate",
        global_config=production_config
    )

PyQt GUI Configuration
---------------------

PyQtGUIConfig
^^^^^^^^^^^^^

.. code-block:: python

    from openhcs.pyqt_gui.config import PyQtGUIConfig, PerformanceMonitorConfig, WindowConfig

    # GUI configuration for PyQt6 interface
    gui_config = PyQtGUIConfig(
        performance_monitor=PerformanceMonitorConfig(
            update_fps=5.0,
            history_duration_seconds=60.0,
            enable_gpu_monitoring=True
        ),
        window=WindowConfig(
            default_width=1200,
            default_height=800,
            remember_window_state=True,
            floating_by_default=True
        ),
        enable_debug_mode=False,
        check_for_updates=True
    )

PerformanceMonitorConfig
^^^^^^^^^^^^^^^^^^^^^^^^

Configuration for system performance monitoring in the GUI:

.. code-block:: python

    monitor_config = PerformanceMonitorConfig(
        update_fps=5.0,                    # Monitor refresh rate
        history_duration_seconds=60.0,     # Data history length
        enable_gpu_monitoring=True,        # GPU metrics
        gpu_temperature_monitoring=True,   # GPU temperature
        cpu_frequency_monitoring=True,     # CPU frequency
        detailed_memory_info=True          # Detailed memory stats
    )

WindowConfig
^^^^^^^^^^^^

Configuration for main window behavior:

.. code-block:: python

    window_config = WindowConfig(
        default_width=1200,
        default_height=800,
        remember_window_state=True,        # Save window state
        floating_by_default=True,          # Non-tiled window
        confirm_close=True,                # Close confirmation
        auto_save_interval_minutes=5       # Auto-save interval
    )

Configuration Enums
-------------------

MaterializationBackend
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: MaterializationBackend
   :members:
   :undoc-members:

Available backends for persistent storage:

- ``ZARR``: Compressed, chunked storage with OME-ZARR metadata
- ``DISK``: Traditional file-based storage

ZarrCompressor
^^^^^^^^^^^^^^

.. autoclass:: ZarrCompressor
   :members:
   :undoc-members:

Available compression algorithms for ZARR storage:

- ``BLOSC``: Fast compression with good ratio
- ``ZLIB``: Standard compression
- ``LZ4``: Very fast compression
- ``ZSTD``: High compression ratio
- ``NONE``: No compression

ZarrChunkStrategy
^^^^^^^^^^^^^^^^^

.. autoclass:: ZarrChunkStrategy
   :members:
   :undoc-members:

Chunking strategies for ZARR arrays:

- ``SINGLE``: Single chunk per array (optimal for batch I/O)
- ``AUTO``: Let ZARR decide chunk size
- ``CUSTOM``: User-defined chunk sizes

See Also
--------

- :doc:`../concepts/pipeline_orchestrator` - Using configuration with orchestrator
- :doc:`../architecture/vfs_system` - VFS configuration details
- :doc:`../concepts/storage_backends` - ZARR configuration examples
- :doc:`orchestrator` - Orchestrator API reference

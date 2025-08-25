Storage System
==============

OpenHCS uses a Virtual File System (VFS) to manage data storage across different backends, enabling efficient processing of large datasets while providing flexibility in where data is stored and how it's accessed.

The Virtual File System (VFS)
-----------------------------

The VFS abstracts storage details so your analysis code works the same regardless of whether data is stored in memory, on disk, or in compressed formats.

.. code-block:: python

   # Your analysis code looks the same regardless of storage backend
   step = FunctionStep(func=process_images, name="process")
   
   # OpenHCS handles storage automatically:
   # - Loads data from appropriate backend
   # - Processes in memory for speed
   # - Saves results to configured backend

**Key principle**: Analysis code is decoupled from storage implementation. You focus on processing logic while OpenHCS manages data movement and storage optimization.

Storage Backends
---------------

OpenHCS provides three main storage backends, each optimized for different use cases:

Memory Backend
~~~~~~~~~~~~~

Stores data in RAM for maximum processing speed.

.. code-block:: python

   from openhcs.constants.constants import Backend
   
   # Data stored in memory (fastest access)
   config.materialization_defaults.backend = Backend.MEMORY

**Characteristics**:
- **Speed**: Fastest access for processing
- **Capacity**: Limited by available RAM
- **Persistence**: Data lost when process ends
- **Use case**: Intermediate processing results, temporary data

**When to use**: For intermediate steps in processing pipelines where speed is critical and data doesn't need to persist.

Disk Backend
~~~~~~~~~~~

Stores data as traditional files on disk storage.

.. code-block:: python

   # Data stored as files on disk (persistent)
   config.materialization_defaults.backend = Backend.DISK

**Characteristics**:
- **Speed**: Moderate access speed (depends on storage type)
- **Capacity**: Large capacity (limited by disk space)
- **Persistence**: Data persists across sessions
- **Use case**: Final results, large datasets, archival storage

**When to use**: For final analysis results, large datasets that exceed memory, or data that needs to persist for future analysis.

Zarr Backend
~~~~~~~~~~~

Stores data in compressed, chunked Zarr format optimized for large arrays.

.. code-block:: python

   # Data stored in compressed Zarr format
   config.materialization_defaults.backend = Backend.ZARR

**Characteristics**:
- **Speed**: Good access speed with compression benefits
- **Capacity**: Very large capacity with efficient compression
- **Persistence**: Data persists with metadata
- **Use case**: Large datasets, long-term storage, data sharing

**When to use**: For large datasets that benefit from compression, data that needs to be shared across platforms, or long-term archival with metadata preservation.

Data Flow During Processing
---------------------------

OpenHCS optimizes data flow to balance speed and storage efficiency:

Typical Processing Flow
~~~~~~~~~~~~~~~~~~~~~~

1. **Input Loading**: Data loaded from source (disk, zarr) into memory
2. **Processing**: All processing happens in memory for speed
3. **Intermediate Storage**: Results stored in memory backend by default
4. **Final Materialization**: Final results saved to configured persistent backend

.. code-block:: text

   Source Data (Disk) → Memory → Processing → Memory → Final Backend
                         ↑                      ↓
                    Fast Access           Configured Storage

Memory-First Strategy
~~~~~~~~~~~~~~~~~~~~

OpenHCS uses a "memory-first" approach for optimal performance:

.. code-block:: python

   # Processing pipeline with automatic memory optimization
   pipeline = Pipeline([
       FunctionStep(func=preprocess, name="preprocess"),    # → Memory
       FunctionStep(func=analyze, name="analyze"),          # → Memory  
       FunctionStep(func=assemble, name="assemble")         # → Final Backend
   ])

**Benefits**:
- **Speed**: Intermediate results accessed from memory
- **Efficiency**: No unnecessary disk I/O during processing
- **Flexibility**: Final storage location configurable

Controlling Output Location
--------------------------

You can control where results are stored at different levels:

Global Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set default storage for all steps
   config.materialization_defaults.backend = Backend.ZARR
   config.materialization_defaults.output_path = "/data/results"

Step-Level Control
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Force specific step output to disk
   step = FunctionStep(
       func=important_analysis,
       name="critical_results",
       force_disk_output=True  # Override global setting
   )

**Use cases for force_disk_output**:
- **Checkpointing**: Save intermediate results for debugging
- **Memory management**: Free memory by persisting large intermediate results
- **Quality control**: Save outputs for manual inspection

Result Organization
------------------

OpenHCS organizes results in a structured directory hierarchy:

Standard Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   /output/plate_name_workspace/
   ├── step1_preprocess/           # Step outputs
   │   ├── A01_s1_processed.tif
   │   ├── A01_s2_processed.tif
   │   └── ...
   ├── step2_analyze/
   │   ├── A01_analysis.csv
   │   ├── A02_analysis.csv  
   │   └── ...
   ├── step3_assemble/
   │   ├── assembled_image.zarr    # Final results
   │   └── metadata.json
   └── openhcs_metadata.json      # Processing provenance

**Directory naming**: Each step creates a subdirectory named after the step, containing all outputs from that processing stage.

Zarr Output Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   /output/results.zarr/
   ├── .zarray                     # Zarr metadata
   ├── .zattrs                     # Custom attributes
   ├── 0.0.0                       # Compressed chunks
   ├── 0.0.1
   ├── 0.1.0
   └── ...

**Benefits of Zarr**:
- **Compression**: Significant space savings for large datasets
- **Chunking**: Efficient access to subsets of large arrays
- **Metadata**: Rich metadata storage with processing history
- **Cross-platform**: Works across different operating systems and languages

Storage Configuration Examples
-----------------------------

VFS Configuration Details
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.core.config import VFSConfig, ZarrConfig, GlobalPipelineConfig
   from openhcs.constants.constants import Backend, MaterializationBackend, ZarrCompressor, ZarrChunkStrategy

   # Complete VFS configuration
   vfs_config = VFSConfig(
       intermediate_backend=Backend.MEMORY,              # Fast memory for intermediate steps
       materialization_backend=MaterializationBackend.ZARR  # ZARR for final results
   )

   # Detailed ZARR configuration
   zarr_config = ZarrConfig(
       store_name="images.zarr",                         # ZARR store filename
       compressor=ZarrCompressor.ZSTD,                   # Compression algorithm
       compression_level=1,                              # Compression level (1-9)
       shuffle=True,                                     # Enable shuffle filter
       chunk_strategy=ZarrChunkStrategy.SINGLE,          # Chunking strategy
       ome_zarr_metadata=True,                           # OME-ZARR metadata
       write_plate_metadata=True                         # Plate-level metadata
   )

   # Integration with global configuration
   global_config = GlobalPipelineConfig(
       vfs=vfs_config,
       zarr=zarr_config,
       num_workers=8
   )

High-Performance Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for speed - keep everything in memory
   config.materialization_defaults.backend = Backend.MEMORY

   # Only save final results to disk
   final_step = FunctionStep(
       func=generate_final_results,
       name="final_results",
       force_disk_output=True
   )

Large Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for large datasets - use compression
   config.materialization_defaults.backend = Backend.ZARR
   config.materialization_defaults.compression_level = 6
   
   # Checkpoint important intermediate results
   checkpoint_step = FunctionStep(
       func=expensive_computation,
       name="checkpoint",
       force_disk_output=True
   )

Mixed Storage Strategy
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Different storage for different types of results
   pipeline = Pipeline([
       # Fast preprocessing in memory
       FunctionStep(func=preprocess, name="preprocess"),
       
       # Save analysis results to disk for inspection
       FunctionStep(
           func=detailed_analysis, 
           name="analysis",
           force_disk_output=True
       ),
       
       # Compress final large results
       FunctionStep(
           func=create_final_dataset,
           name="final",
           backend=Backend.ZARR
       )
   ])

Storage Performance Considerations
---------------------------------

Memory Usage
~~~~~~~~~~~

**Memory requirements**: Processing large datasets requires sufficient RAM for intermediate results.

**Memory optimization**: Use ``force_disk_output=True`` for large intermediate results to free memory for subsequent processing.

I/O Performance
~~~~~~~~~~~~~~

**SSD vs HDD**: SSD storage provides significantly better performance for disk backend operations.

**Network storage**: Network-attached storage may be slower but provides better capacity and sharing capabilities.

**Compression trade-offs**: Zarr compression reduces storage space but requires CPU time for compression/decompression.

Backend Selection Guidelines
----------------------------

**Use Memory Backend When**:
- Processing small to medium datasets (fits in RAM)
- Need maximum processing speed
- Working with temporary intermediate results

**Use Disk Backend When**:
- Need persistent storage of results
- Working with datasets larger than available RAM
- Require compatibility with external tools

**Use Zarr Backend When**:
- Processing very large datasets
- Need efficient compression
- Require rich metadata storage
- Sharing data across platforms

Directory Organization and Workspaces
------------------------------------

OpenHCS uses a Virtual File System (VFS) approach that primarily operates on data in memory, creating physical directories only when necessary.

Workspace Creation and Data Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Workspace Creation**: OpenHCS creates a symlinked mirror of input data for safe, isolated processing
2. **In-Memory Operations**: Processing occurs entirely in memory between steps
3. **Final Materialization**: Results written to persistent storage only at pipeline completion

.. code-block:: text

   # Input (read-only)
   /path/to/input_data/plate1/
   ├── Images/
   │   └── ... (image files)
   └── Metadata/
       └── ... (metadata files)

   # Output (generated by OpenHCS)
   /path/to/my_hcs_results/
   └── plate1_processed/              # {plate_name}{output_dir_suffix}
       ├── images.zarr/             # Final image data
       └── analysis/                # Analysis results
           └── cell_counts.csv

Configuration-Driven Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Directory structure is controlled by ``PathPlanningConfig``:

.. code-block:: python

   from openhcs.core.config import PathPlanningConfig

   config = GlobalPipelineConfig(
       path_planning=PathPlanningConfig(
           global_output_folder="/path/to/results/",
           output_dir_suffix="_processed",
           materialization_results_path="analysis"
       )
   )

**Parameters**:
- **global_output_folder**: Top-level directory for all plate outputs
- **output_dir_suffix**: Suffix for final materialized image data directory
- **materialization_results_path**: Subdirectory for non-image results (CSV, analysis files)

The storage system provides flexible, efficient data management that scales from small experiments to large-scale high-content screening datasets while maintaining consistent interfaces for analysis code.

Storage and Memory System Architecture
======================================

Overview
--------

OpenHCS implements a unified storage and memory system that addresses common challenges of scientific image processing: how to efficiently handle datasets that range from small test images to large experimental plates while maintaining performance, type safety, and integration across different computational backends.

**The Scientific Computing Challenge**: Traditional image analysis tools can struggle with large high-content screening datasets. A typical experiment might generate many individual TIFF files totaling substantial amounts of data. Common issues include out-of-memory errors, slow processing times, or data corruption during format conversions.

**The OpenHCS Solution**: A two-layer architecture that combines a Virtual File System (VFS) for storage abstraction with a Memory Type System for computational backend management. This allows the same code to work whether processing small test images or large experimental datasets, while automatically handling conversions between NumPy, PyTorch, CuPy, TensorFlow, JAX, and pyclesperanto arrays.

**Key Feature**: The system treats storage backends and memory types as orthogonal concerns. You can store data in memory, on disk, or in compressed ZARR format regardless of whether it's a NumPy array, PyTorch tensor, or CuPy array. This separation enables optimization strategies that would be difficult with tightly coupled systems.

Virtual File System (VFS) Architecture
---------------------------------------

The VFS layer addresses the "storage backend explosion" problem common in scientific computing: different tools have their own preferred storage formats, leading to many conversion utilities and format compatibility issues.

**The Problem**: Scientific workflows often involve multiple tools, each with different storage preferences. ImageJ works with TIFF files, deep learning frameworks prefer HDF5 or custom formats, and analysis tools often require CSV or JSON outputs. Managing these format differences manually leads to brittle pipelines and data corruption risks.

**The Solution**: A unified abstraction layer that provides location-transparent data access across different storage backends. The same logical path works whether data is stored in memory for speed, on disk for persistence, or in compressed ZARR format for large datasets.

Backend Abstraction
~~~~~~~~~~~~~~~~~~~~

The VFS abstracts away the underlying storage mechanism through a common interface that hides complexity while enabling optimization:

.. code:: python

   # Same API regardless of where data is stored
   filemanager.save(data, "path/to/data", Backend.MEMORY)
   filemanager.save(data, "path/to/data", Backend.DISK)
   filemanager.save(data, "path/to/data", Backend.ZARR)

   # Load from any backend
   data = filemanager.load("path/to/data", Backend.MEMORY)
   data = filemanager.load("path/to/data", Backend.DISK)

**Why This Matters**: The same processing code works regardless of where data is stored. During development, you might use the memory backend for speed. For production runs, you might use the disk backend for reliability. For large datasets, you might use the ZARR backend for compression. The processing logic never changes.

Backend Type Hierarchy
~~~~~~~~~~~~~~~~~~~~~~

OpenHCS defines a hierarchy of backend abstractions to support different storage paradigms:

.. code:: python

   # Base interface for all data destinations
   class DataSink(ABC):
       """Minimal interface for sending data to any destination."""

       @abstractmethod
       def save(self, data: Any, identifier: Union[str, Path], **kwargs) -> None:
           """Send data to the destination."""
           pass

       @abstractmethod
       def save_batch(self, data_list: List[Any], identifiers: List[Union[str, Path]], **kwargs) -> None:
           """Send multiple data objects in a single operation."""
           pass

   # Backends with real filesystems (disk, memory, zarr)
   class StorageBackend(DataSink):
       """Persistent storage with file-like semantics and retrieval capabilities."""

       @abstractmethod
       def load(self, file_path: Union[str, Path], **kwargs) -> Any:
           """Load data from storage."""
           pass

   # Backends without real filesystems (OMERO, cloud storage)
   class VirtualBackend(DataSink):
       """Virtual filesystem semantics - generates file listings on-demand."""

       @abstractmethod
       def list_files(self, directory: str, **kwargs) -> List[str]:
           """Generate file list from metadata (not real filesystem)."""
           pass

       @abstractmethod
       def generate_filename(self, metadata: Dict, **kwargs) -> str:
           """Generate filename from metadata."""
           pass

**Key Distinction**: ``StorageBackend`` provides traditional file operations (load/save), while ``VirtualBackend`` generates filenames on-demand from metadata without real filesystem operations. This enables location-transparent processing where the same pipeline code works on disk, zarr, or OMERO without modification.

**Examples**:

- **StorageBackend**: DiskStorageBackend, MemoryStorageBackend, ZarrStorageBackend
- **VirtualBackend**: OMEROLocalBackend (generates paths from OMERO plate structure), VirtualWorkspaceBackend (metadata-based path mapping)
- **StreamingBackend**: NapariStreamBackend, FijiStreamBackend (real-time visualization)

See :doc:`omero_backend_system` for detailed VirtualBackend architecture and OMERO integration.

Virtual Workspace Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Metadata-based workspace initialization without physical file operations

**When to Use**: Microscope formats with complex directory structures (TimePoint folders, ZStep folders) that need flattening for processing.

**The Problem**: Some microscope formats (ImageXpress, Opera Phenix) organize images in nested folders (e.g., ``TimePoint_1/ZStep_2/image.tif``). Traditional workspace preparation creates physical symlinks or copies files to flatten the structure, which is slow and wastes disk space.

**The Solution**: VirtualWorkspaceBackend stores a plate-relative path mapping in ``openhcs_metadata.json`` that translates virtual flattened paths to real nested paths without any physical file operations.

**Architecture**:

.. code-block:: python

   # Metadata-based mapping (stored in openhcs_metadata.json)
   {
     "workspace_mapping": {
       "images/A01_s001_w1_z001_t001.tif": "TimePoint_1/ZStep_1/A01_s001_w1.tif",
       "images/A01_s001_w1_z001_t002.tif": "TimePoint_2/ZStep_1/A01_s001_w1.tif"
     }
   }

   # VirtualWorkspaceBackend translates paths on-the-fly
   backend.load("images/A01_s001_w1_z001_t001.tif")
   # → Actually loads "TimePoint_1/ZStep_1/A01_s001_w1.tif"

**Integration with Microscope Handlers**:

Microscope handlers that need virtual workspace mapping (e.g., ImageXpress, Opera Phenix) implement ``_build_virtual_mapping()`` to generate the workspace mapping. This method is **optional** and only needed for handlers that use the base class ``initialize_workspace()`` implementation.

Handlers that override ``initialize_workspace()`` completely (like OMERO and OpenHCS) don't need to implement this method because they handle workspace initialization differently:

- **OMERO**: Uses database-backed virtual filesystem, doesn't need file-based mapping
- **OpenHCS**: Already uses normalized format, doesn't need directory flattening

**Example Implementation** (ImageXpress, Opera Phenix):

.. code-block:: python

   def _build_virtual_mapping(self, plate_path: Path, filemanager: FileManager) -> Path:
       """Build virtual workspace mapping for nested folder structures."""
       workspace_mapping = {}

       # Scan for TimePoint/ZStep folders and build mappings
       for subdir in subdirs:
           self._flatten_timepoints(subdir, filemanager, workspace_mapping, plate_path)
           self._flatten_zsteps(subdir, filemanager, workspace_mapping, plate_path)

       # Save mapping to metadata
       metadata_writer.update_metadata(
           plate_path,
           {"workspace_mapping": workspace_mapping}
       )

       return plate_path

**Backend Selection and Preference Hierarchy**:

OpenHCS uses a strict backend preference hierarchy when multiple backends are available:

**Preference Order**: ``zarr`` > ``virtual_workspace`` > ``disk``

This hierarchy ensures that:

1. **Zarr backend** is preferred when available (optimized for large datasets, compressed storage)
2. **Virtual workspace** is used for original nested microscope data (when zarr not available)
3. **Disk backend** is the fallback (standard TIFF files)

**Main Subdirectory Selection**:

When loading metadata with multiple subdirectories, OpenHCS selects the subdirectory marked with ``"main": true``:

.. code-block:: json

   {
     "subdirectories": {
       ".": {
         "workspace_mapping": {...},
         "available_backends": {"disk": true, "virtual_workspace": true},
         "main": false
       },
       "zarr": {
         "image_files": [...],
         "available_backends": {"zarr": true},
         "main": true
       }
     }
   }

The main subdirectory determines:

- Which backend is used by ``get_primary_backend()``
- Which image list is returned by ``get_image_files()``
- Which metadata is used for pipeline execution

**Automatic Backend Selection**:

The virtual workspace backend is automatically selected when:

1. Microscope handler's ``get_available_backends()`` detects ``workspace_mapping`` in metadata
2. FileManager registers VirtualWorkspaceBackend in local registry
3. ``get_primary_backend()`` returns ``"virtual_workspace"`` for reading original data (if zarr not available)

**Metadata emission (centralized helper)**:

- Virtual workspace handlers now use a shared helper to write ``openhcs_metadata.json`` so all metadata fields stay consistent.
- The helper records the workspace mapping plus handler/parser names and optional grid/pixel metadata if available.
- Orchestrator metadata extraction respects the handler's chosen primary backend (e.g., virtual workspace for ImageXpress/Opera), instead of forcing ``disk``.

**Materialization Compatibility**:

Virtual workspace works seamlessly with materialization backends:

- **Read backend**: ``virtual_workspace`` (for original nested data)
- **Write backend**: ``disk`` or ``zarr`` (for flattened outputs)

This enables processing pipelines to read from nested structures while writing to flattened outputs without any physical workspace preparation.

**Performance**:

- **Initialization**: ~100ms to build mapping for 1000 files (vs ~10s for symlink creation)
- **Runtime overhead**: Negligible (simple dictionary lookup per file access)
- **Disk space**: Zero overhead (no symlinks or copies)

**Supported Microscope Formats**:

- **ImageXpress**: TimePoint and ZStep folder flattening
- **Opera Phenix**: Nested field/plane folder flattening

See :doc:`microscope_handler_integration` for microscope-specific implementation details.

Path Virtualization
~~~~~~~~~~~~~~~~~~~

VFS provides a unified path interface where the same logical path works across all backends:

-  **Unified Path**: ``/pipeline/step1/output/processed_images``
-  **Memory Backend**: Stores in-memory using the same path as key
-  **Disk Backend**: Maps to physical file using the same path structure
-  **Zarr Backend**: Creates zarr store using the same path structure

**The Design Philosophy**: Paths are identical across all backends - the VFS handles the backend-specific storage implementation transparently. This means you can switch storage strategies without changing any processing code.

Storage Backend Types
~~~~~~~~~~~~~~~~~~~~~

Memory Backend
^^^^^^^^^^^^^^

**Purpose**: Fast intermediate data storage for processing pipelines

**When to Use**: Temporary arrays and tensors between pipeline steps where speed is critical and persistence isn't required.

**Characteristics**:
- Fast access (direct object access)
- Limited by available RAM
- Volatile (lost on process exit)
- Supports any Python object

**Real-World Usage**: In image processing pipelines, intermediate results like filtered images or segmentation masks are stored in the memory backend for speed.

**Materialization Integration**: When steps need to save additional outputs (like analysis results), the memory backend serves as the staging area before materialization to persistent storage.

Disk Backend
^^^^^^^^^^^^

**Purpose**: Persistent data storage with standard file format support

**When to Use**: Input images, final outputs, checkpoints, and any data that needs to survive process restarts.

**Characteristics**:
- Persistent across runs
- Slower than memory but faster than network storage
- Unlimited capacity (limited only by disk space)
- Supports standard file formats (TIFF, PNG, NPY, etc.)

**Real-World Usage**: Original microscopy images are loaded from the disk backend, and final analysis results are saved back to disk for long-term storage.

Zarr Backend
^^^^^^^^^^^^

**Purpose**: Chunked array storage with OME-ZARR support for large datasets

**When to Use**: Large multidimensional arrays, compressed storage, and datasets that need to be accessed from multiple tools.

**Characteristics**:
- Efficient for large arrays
- Supports compression (ZSTD, LZ4) with significant size reduction
- Cloud storage compatible
- OME-ZARR metadata support for interoperability
- Parallel access for multi-threaded processing

**Real-World Usage**: Final processed datasets from high-content screening experiments are stored in ZARR format for sharing and analysis.

Memory Type System
------------------

The Memory Type System addresses the "computational backend fragmentation" problem in scientific Python: different libraries use different array types, leading to conversion overhead and compatibility issues.

**The Problem**: Modern scientific computing involves multiple specialized libraries. NumPy provides the foundation, but PyTorch is used for deep learning, CuPy for GPU acceleration, pyclesperanto for image processing, and JAX for high-performance computing. Each library has its own array type, and converting between them can be error-prone and slow.

**The Solution**: A unified memory type system that handles conversions automatically while maintaining type safety and performance. The system knows how to convert between supported array types and can optimize conversions to minimize data copying.

Supported Memory Types
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Memory Type Support Matrix
   :header-rows: 1
   :widths: 20 20 15 25 20

   * - Memory Type
     - Library
     - GPU Support
     - Use Cases
     - Image Serialization
   * - ``numpy``
     - NumPy
     - No
     - CPU processing, I/O operations
     - ``.tiff`` (disk), zarr chunks (zarr)
   * - ``cupy``
     - CuPy
     - Yes
     - GPU-accelerated NumPy-like operations
     - ``.tiff`` (disk), zarr chunks (zarr)
   * - ``torch``
     - PyTorch
     - Yes
     - Deep learning, neural networks
     - ``.tiff`` (disk), zarr chunks (zarr)
   * - ``tensorflow``
     - TensorFlow
     - Yes
     - Machine learning, TensorFlow models
     - ``.tiff`` (disk), zarr chunks (zarr)
   * - ``jax``
     - JAX
     - Yes
     - High-performance computing, research
     - ``.tiff`` (disk), zarr chunks (zarr)
   * - ``pyclesperanto``
     - pyclesperanto
     - Yes
     - GPU-accelerated image processing
     - ``.tiff`` (disk), zarr chunks (zarr)

**Important Note**: Regardless of the memory type used during processing, all image data is converted to NumPy arrays for serialization. The disk backend saves images as standard TIFF files, while the zarr backend saves them as compressed zarr chunks. The memory type only affects computational processing, not storage format.

**Design Principle**: Each memory type is optimized for specific use cases, but the conversion system ensures they can all work together seamlessly.

Automatic Type Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~

The system implements intelligent conversion strategies that minimize performance overhead. When functions require specific memory types, the system automatically converts between them while preserving data integrity.

**Conversion Strategy**: The memory type system detects the required input type for each function and converts the data accordingly. After processing, the result maintains the target memory type for subsequent operations.

**Example Conversion Flow** (conceptual illustration):

.. code:: python

   # Conceptual example - not actual OpenHCS API
   # Step 1: Load TIFF → process with CuPy function
   # System converts numpy → cupy automatically

   # Step 2: Process with PyTorch function
   # System converts cupy → torch automatically

   # Step 3: Process with NumPy function → save to disk
   # System converts torch → numpy automatically

**Real OpenHCS Pipeline Example**:

.. code:: python

   # Actual OpenHCS FunctionStep API
   pipeline = [
       FunctionStep(func="gaussian_filter", sigma=2.0),      # CuPy function
       FunctionStep(func="threshold_otsu"),                  # scikit-image function
       FunctionStep(func="binary_opening", footprint=disk(3)) # CuPy function
   ]

**Conversion Optimization**: The system uses zero-copy transfers where possible (like CuPy ↔ PyTorch via DLPack) and reduces CPU-GPU transfers by keeping data on the GPU when consecutive operations support it.

Memory Type Declaration System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS functions declare their memory interface using decorators that enable automatic type conversion and GPU memory management. This system enforces explicit memory type contracts while providing automatic optimization.

**Basic Memory Type Decorators**:

.. code:: python

   from openhcs.core.memory.decorators import numpy, cupy, torch, jax, pyclesperanto

   @numpy
   def process_cpu(image_stack):
       """CPU processing with NumPy arrays."""
       import numpy as np
       return np.median(image_stack, axis=0, keepdims=True)

   @cupy
   def process_gpu_cupy(image_stack):
       """GPU processing with CuPy arrays."""
       import cupy as cp
       return cp.median(image_stack, axis=0, keepdims=True)

   @torch(oom_recovery=True)
   def process_gpu_torch(image_stack):
       """GPU processing with PyTorch tensors and automatic OOM recovery."""
       import torch
       return torch.median(image_stack, dim=0, keepdim=True)[0]

   @pyclesperanto(oom_recovery=True)
   def process_gpu_opencl(image_stack):
       """GPU processing with pyclesperanto OpenCL arrays."""
       import pyclesperanto_prototype as cle
       return cle.median_sphere(image_stack, radius_x=1, radius_y=1, radius_z=0)

**Advanced Memory Type Specification**:

.. code:: python

   from openhcs.core.memory.decorators import memory_types

   # Mixed input/output types
   @memory_types(input_type="numpy", output_type="torch")
   def neural_network_inference(image_stack):
       """Convert NumPy input to PyTorch for GPU inference."""
       import torch
       # Function receives NumPy array, returns PyTorch tensor
       model = torch.load('model.pt')
       return model(image_stack)

   # Explicit type specification with custom settings
   @torch(input_type="torch", output_type="torch", oom_recovery=True)
   def memory_intensive_operation(image_stack):
       """GPU operation with automatic OOM recovery."""
       # Automatic GPU memory management and thread-local CUDA streams
       return torch.nn.functional.conv3d(image_stack, kernel)

**Automatic Features**:

- **Thread-Local CUDA Streams**: Each thread gets persistent CUDA streams for true parallelization
- **OOM Recovery**: Automatic out-of-memory recovery with CPU fallback
- **Device Management**: Automatic GPU device placement and management
- **Type Validation**: Runtime validation of input/output memory types

Stack/Unstack Operations
~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS enforces a 3D array discipline to prevent shape-related bugs that are common in image processing pipelines:

**The Problem**: Scientific image processing often involves mixing 2D and 3D operations, leading to shape mismatches and silent failures. A function expecting a 3D stack might receive a 2D image, or vice versa.

**The Solution**: All functions must operate on 3D arrays of shape ``[Z, Y, X]``, even for single 2D images. The stack/unstack system handles conversions between 2D image lists and 3D arrays while maintaining type safety.

**Stack/Unstack API** (conceptual - actual implementation may vary):

.. code:: python

   # Conceptual example of stack/unstack operations
   # Convert list of 2D images to 3D array with specified memory type
   stack_3d = stack_slices(
       slices=[img1_2d, img2_2d, img3_2d],  # List of 2D arrays (any memory type)
       memory_type="torch",                  # Target memory type
       gpu_id=0                             # GPU device ID
   )
   # Returns: torch.Tensor of shape [3, Y, X] on GPU 0

   # Convert 3D array back to list of 2D slices
   slices_2d = unstack_slices(
       array=stack_3d,        # 3D array (any memory type)
       memory_type="numpy",   # Target memory type for output slices
       gpu_id=0              # GPU device ID
   )

**Validation Benefits**: This approach catches shape errors at the boundary between 2D and 3D operations, preventing silent failures that could corrupt scientific results.

**Real-World Usage**: In practice, the OpenHCS pipeline automatically handles stacking and unstacking as images flow between processing steps, ensuring consistent 3D array format throughout the pipeline.

Memory Conversion System
~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS implements a comprehensive memory conversion system that enables seamless transitions between different array libraries while maintaining GPU efficiency and data integrity.

**Conversion Architecture**:

.. code:: python

   # Zero-copy conversions (preferred)
   def _cupy_to_torch_dlpack(data, device_id=None):
       """Convert CuPy to PyTorch using DLPack (zero-copy)."""
       import torch
       dlpack = data.toDlpack()
       return torch.from_dlpack(dlpack)

   def _torch_to_cupy_dlpack(data, device_id=None):
       """Convert PyTorch to CuPy using DLPack (zero-copy)."""
       import cupy as cp
       dlpack = data.__dlpack__()
       return cp.fromDlpack(dlpack)

   # CUDA Array Interface conversions
   def _cupy_to_pyclesperanto_cuda(data, device_id=None):
       """Convert CuPy to pyclesperanto using CUDA Array Interface."""
       import pyclesperanto_prototype as cle
       cle.select_device(device_id or 0)
       return cle.asarray(data)  # Uses __cuda_array_interface__

**Conversion Strategy Hierarchy**:

1. **Zero-Copy GPU-to-GPU**: DLPack, CUDA Array Interface (preferred)
2. **CPU Roundtrip**: Fallback when direct GPU conversion fails
3. **Error Handling**: Detailed error reporting with conversion context

.. code:: python

   class MemoryConversionError(Exception):
       """Raised when memory type conversion fails."""
       def __init__(self, source_type: str, target_type: str, method: str, reason: str):
           self.source_type = source_type
           self.target_type = target_type
           self.method = method
           self.reason = reason
           super().__init__(f"Failed to convert {source_type} → {target_type} via {method}: {reason}")

**GPU Memory Management**: The conversion system integrates with OpenHCS GPU cleanup utilities:

.. code:: python

   def cleanup_gpu_memory_by_framework(memory_type: str, device_id: Optional[int] = None):
       """Clean up GPU memory based on OpenHCS memory type."""
       if memory_type == "torch":
           cleanup_pytorch_gpu(device_id)
       elif memory_type == "cupy":
           cleanup_cupy_gpu(device_id)
       elif memory_type == "pyclesperanto":
           cleanup_pyclesperanto_gpu(device_id)
       # ... other frameworks

   def cleanup_all_gpu_frameworks(device_id: Optional[int] = None):
       """Comprehensive GPU cleanup for all frameworks."""
       cleanup_pytorch_gpu(device_id)
       cleanup_cupy_gpu(device_id)
       cleanup_tensorflow_gpu(device_id)
       cleanup_jax_gpu(device_id)
       cleanup_pyclesperanto_gpu(device_id)

**Conversion Performance**: The system prioritizes GPU-to-GPU transfers and minimizes CPU roundtrips, achieving near-zero overhead for compatible memory types.

Materialization System
----------------------

The materialization system bridges the gap between computational processing and persistent storage. It handles the conversion of function side effects (analysis results, metadata, derived data) from memory backend staging to persistent storage in appropriate formats.

**The Problem**: Scientific image processing functions often produce valuable side effects beyond the main image output - cell counts, position coordinates, analysis metrics, segmentation masks. These need to be saved in formats that researchers can use with standard analysis tools (CSV, JSON, TIFF), but the computational functions work with Python objects in memory.

**The Solution**: A materialization system that automatically converts function side effects to appropriate file formats and saves them using the storage backend system. This provides a clean separation between computational logic and storage concerns.

Special Output Decoration
~~~~~~~~~~~~~~~~~~~~~~~~~

Functions declare their side effects using the ``@special_outputs`` decorator, which can optionally specify materialization specs (resolved via format writers) for converting data to persistent formats.

**Basic Special Outputs** (memory backend only):

.. code:: python

   from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs

   @special_outputs("positions", "metadata")
   def generate_positions(image_stack):
       """Function that produces special outputs stored in memory."""
       positions = calculate_positions(image_stack)
       metadata = extract_metadata(image_stack)

       # Return: (main_output, special_output_1, special_output_2, ...)
       return processed_image, positions, metadata

**With Materialization Specs** (memory + persistent storage):

.. code:: python

   from openhcs.processing.materialization import MaterializationSpec, CsvOptions, ROIOptions

   @special_outputs(("cell_counts", MaterializationSpec(CsvOptions(filename_suffix="_details.csv"))),
                     ("masks", MaterializationSpec(ROIOptions())))
   def count_cells_with_materialization(image_stack):
       """Function with materialized special outputs."""
       processed_image, cell_counts, segmentation_masks = analyze_cells(image_stack)

       # cell_counts and masks are automatically materialized to disk
       return processed_image, cell_counts, segmentation_masks

**Mixed Declaration** (some materialized, some memory-only):

.. code:: python

   from openhcs.processing.materialization import MaterializationSpec, CsvOptions

   @special_outputs("debug_info", ("analysis_results", MaterializationSpec(CsvOptions(filename_suffix=".csv"))))
   def analyze_with_mixed_outputs(image_stack):
       """Function with both memory-only and materialized outputs."""
       # debug_info stays in memory, analysis_results gets materialized
       return processed_image, debug_info, analysis_results

Note: Writer dispatch is automatically inferred from options type. No need to specify handler strings.
The ``fields`` parameter is optional and defaults to None (auto-extract all fields).
Only specify ``fields`` when you need to control column ordering or select a subset.

Writer-Based Materialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Materialization is implemented as a small set of format writers (CSV/JSON/ROI ZIP/TIFF/TEXT). Call sites never register custom materializers.

Instead, functions declare what they want persisted by providing writer options to ``MaterializationSpec``.

.. code:: python

   from openhcs.processing.materialization import MaterializationSpec, CsvOptions, JsonOptions

   # JSON summary + CSV details, JSON is primary
   spec = MaterializationSpec(
       JsonOptions(source="summary", filename_suffix=".json"),
       CsvOptions(source="details", filename_suffix="_details.csv"),
       primary=0,
   )

The framework handles:

- path normalization (including compound suffix stripping)
- directory creation and overwrite/delete semantics
- multi-backend iteration

       unique_x = sorted(df['x_position_um'].unique())
       unique_y = sorted(df['y_position_um'].unique())
       df['grid_dimensions'] = f"{len(unique_y)}x{len(unique_x)}"
       df['algorithm'] = 'ashlar_cpu'

       # Save using FileManager
       csv_content = df.to_csv(index=False)
       filemanager.save(csv_content, csv_path, "disk")
       return csv_path

Configuration Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

The materialization system integrates with the OpenHCS configuration hierarchy to control when and how materialization occurs. Configuration settings determine materialization behavior at multiple levels.

**Configuration Integration**: The materialization system integrates with the OpenHCS configuration hierarchy through several key configuration objects:

- **VFS Configuration**: Controls backend selection (memory for intermediate, disk/zarr for materialization)
- **Step Materialization Configuration**: Controls per-step materialization behavior and directory naming
- **Configuration Resolution**: Follows the standard OpenHCS hierarchy (step → pipeline → global)

For complete configuration details and examples, see :doc:`configuration_framework`.

**Architectural Pattern**: The configuration system provides declarative control over materialization behavior without requiring code changes. The same materialization handler can save to different backends based purely on configuration settings.

Execution Flow
~~~~~~~~~~~~~~

The materialization system operates in two phases during pipeline execution: memory staging and persistent materialization.

**Phase 1: Memory Staging** (automatic):

1. **Function Execution**: Function runs and produces main output + special outputs
2. **Memory Storage**: Special outputs are automatically saved to memory backend using VFS paths
3. **Path Planning**: Compiler creates VFS paths for special outputs during compilation
4. **Cross-Step Access**: Other steps can load special outputs from memory backend using ``@special_inputs``

**Phase 2: Persistent Materialization** (conditional):

1. **Materialization Check**: System checks if special output has associated materialization spec
2. **Data Loading**: Loads special output data from memory backend
3. **Format Conversion**: Materialization handler converts data to appropriate file format
4. **Backend Storage**: Saves converted data using configured materialization backend (disk/zarr)
5. **Path Return**: Returns path to materialized file for logging/reference

**Execution Example**:

.. code:: python

   # During pipeline execution:

   # 1. Function executes
   @special_outputs(("cell_counts", materialize_cell_counts))
   def count_cells(image_stack):
       return processed_image, cell_count_results

   # 2. Automatic memory staging
   # - processed_image → memory backend (standard pipeline flow)
   # - cell_count_results → memory backend at VFS path "/memory/step_output/cell_counts.pkl"

   # 3. Materialization execution (if materialization spec exists)
   # - Load cell_count_results from memory backend
   # - Call materialize_cell_counts(cell_count_results, "/memory/step_output/cell_counts.pkl", filemanager)
   # - Save CSV/JSON files to disk backend
   # - Log materialization completion

**Configuration-Driven Behavior**: The materialization backend (disk vs zarr) is determined by the ``VFSConfig.materialization_backend`` setting, allowing the same materialization handler to save to different storage formats based on configuration.

System Integration Patterns
---------------------------

The storage and memory systems work together to provide seamless data flow through complex processing pipelines.

**The Integration Challenge**: How do you coordinate storage decisions (where to put data) with memory type decisions (what format to use) without creating tight coupling between the systems?

**The Solution**: The systems are designed as orthogonal layers that can be combined independently. Storage backends handle persistence and location, while memory types handle computational format and device placement.

VFS + Memory Type Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During pipeline execution, the systems coordinate to optimize data flow. The VFS manages storage locations while the memory type system handles computational formats.

**Standard Pipeline Flow**: OpenHCS uses a consistent pattern where intermediate results are stored in the memory backend for speed, while final results are materialized to either disk or zarr backend based on user preference.

**Real OpenHCS Pipeline Example**:

.. code:: python

   # Actual OpenHCS FunctionStep API
   pipeline = [
       # Step 1: Load from disk → process → store in memory
       FunctionStep(func="gaussian_filter", sigma=2.0),

       # Step 2: Load from memory → GPU processing → store in memory
       FunctionStep(func="binary_opening", footprint=disk(3)),

       # Step 3: Load from memory → process → materialize to disk/zarr
       FunctionStep(func="label", connectivity=2)
   ]

**Backend Usage Pattern**:
- **Input**: Always loaded from disk backend (original TIFF files)
- **Intermediate**: Always stored in memory backend for speed
- **Output**: Materialized to disk backend (.tiff files) or zarr backend (compressed chunks)
- **Special Outputs**: Staged in memory backend, optionally materialized to persistent storage

**Coordination Benefits**: The VFS handles where data lives, the memory type system handles what format it's in, and the materialization system handles conversion to persistent formats. The integration layer coordinates between all three to minimize unnecessary conversions and data movement.

FileManager Advanced Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FileManager provides comprehensive file and directory operations beyond basic save/load functionality, with automatic backend selection and natural sorting integration.

**Batch Operations**:

.. code:: python

   # Batch loading for performance
   file_paths = ["/memory/image_001.pkl", "/memory/image_002.pkl", "/memory/image_003.pkl"]
   batch_data = filemanager.load_batch(file_paths, Backend.MEMORY.value)

   # Batch saving with backend-specific optimizations
   image_arrays = [array1, array2, array3]
   output_paths = ["/zarr/output_001", "/zarr/output_002", "/zarr/output_003"]
   filemanager.save_batch(image_arrays, output_paths, Backend.ZARR.value,
                         chunk_name="well_A01", zarr_config=zarr_config)

**Directory Operations**:

.. code:: python

   # List files with natural sorting (handles numeric sequences correctly)
   image_files = filemanager.list_image_files("/disk/plate/", Backend.DISK.value,
                                             extensions=['.tif', '.tiff'],
                                             recursive=True)
   # Returns: ['image_001.tif', 'image_002.tif', ..., 'image_010.tif'] (not lexicographic)

   # Directory listing with metadata
   entries = filemanager.list_dir("/memory/step_outputs/", Backend.MEMORY.value)

   # Directory mirroring with symlinks
   filemanager.mirror_directory("/disk/source/", "/disk/target/", Backend.DISK.value,
                               overwrite_symlinks_only=True)

**Advanced File Operations**:

.. code:: python

   # Atomic file operations
   filemanager.move("/disk/temp/file.tif", "/disk/final/file.tif", Backend.DISK.value,
                   replace_symlinks=False)

   # Directory creation with backend validation
   filemanager.ensure_directory("/zarr/new_experiment/", Backend.ZARR.value)

   # File existence checking across backends
   exists = filemanager.exists("/memory/intermediate_result.pkl", Backend.MEMORY.value)

**Storage Registry System**: The FileManager uses a metaclass-based auto-registration system for backend management:

.. code:: python

   # Backends are automatically registered when their classes are defined
   from polystore.backend_registry import StorageBackendMeta, STORAGE_BACKENDS, create_storage_registry
   from polystore.base import DataSink

   # Example backend with automatic registration
   class DiskStorageBackend(StorageBackend, metaclass=StorageBackendMeta):
       """Disk storage backend with automatic metaclass registration."""

       # Backend type from enum for registration
       _backend_type = Backend.DISK.value

       def __init__(self):
           # Backend initialization
           pass

   # The metaclass automatically registers the backend in STORAGE_BACKENDS
   # STORAGE_BACKENDS: Dict[str, Type[DataSink]] = {}  # Populated by metaclass

   # Create registry with all discovered backends
   def create_storage_registry() -> Dict[str, DataSink]:
       """
       Create storage registry with all registered backends.

       Returns:
           Dictionary mapping backend types to instances
       """
       # Ensure all backends are discovered
       discover_all_backends()

       registry = {}
       for backend_type in STORAGE_BACKENDS.keys():
           registry[backend_type] = get_backend_instance(backend_type)

       return registry

   # Global singleton storage registry
   from polystore.base import storage_registry  # Created at module import

**Lazy Backend Discovery**: The backend discovery system uses lazy imports to avoid loading GPU-heavy backends during startup:

.. code:: python

   # openhcs/io/__init__.py
   def __getattr__(name: str):
       """Lazy import for GPU-heavy backends."""
       if name in ['zarr', 'napari_stream', 'fiji_stream']:
           # Return the class without importing the module
           # Module import happens later during discover_all_backends()
           return getattr(sys.modules[__name__], name)
       raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

   # Backend registration happens during module import via metaclass
   def discover_all_backends() -> None:
       """Discover and register all available backends.

       Uses importlib.import_module() to directly import GPU-heavy backend modules,
       ensuring metaclass registration happens even with lazy imports.
       """
       # Import GPU-heavy backends directly from their modules (not via __getattr__)
       # This ensures the module is imported and metaclass registration happens
       importlib.import_module('polystore.zarr')
       importlib.import_module('polystore.napari_stream')
       importlib.import_module('polystore.fiji_stream')

**Key Design Points**:

- **Lazy Loading**: GPU-heavy backends (zarr, napari_stream, fiji_stream) are not imported during ``polystore`` module load
- **Explicit Discovery**: ``discover_all_backends()`` uses ``importlib.import_module()`` to trigger module imports
- **Metaclass Registration**: Backend classes register themselves via ``StorageBackendMeta`` during module import
- **Startup Performance**: Defers GPU library imports (ome-zarr, napari, fiji) until first use

**Subprocess Runner Mode**: The backend discovery system supports a special subprocess runner mode that avoids importing GPU-heavy backends:

.. code:: python

   import os

   # Check if we're in subprocess runner mode
   if os.getenv('OPENHCS_SUBPROCESS_NO_GPU') == '1':
       # Subprocess runner mode - only import essential backends
       from polystore import disk, memory
       # Skips: zarr, napari_stream, fiji_stream (avoid GPU library imports)
   else:
       # Normal mode - discover all backends (including GPU-heavy ones)
       from polystore.backend_registry import discover_all_backends
       discover_all_backends()

This prevents GPU library imports in subprocess workers, which is critical for multiprocessing execution where worker processes don't need GPU access.

**Natural Sorting Integration**: All file listing operations use natural sorting to handle numeric sequences correctly, preventing issues with lexicographic ordering of scientific image sequences.

Performance Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integrated system enables various optimization strategies based on the standard OpenHCS backend usage pattern:

**Standard Backend Strategy**:
- **Memory Backend**: Always used for intermediate results during pipeline execution
- **Materialization Choice**: Disk vs Zarr backend for final outputs based on use case

**Backend Selection Criteria**:

1. **Disk Backend (Standard TIFF)**:
   - Easy to use with standard image analysis tools
   - Compatible with ImageJ, napari, Fiji out-of-the-box
   - Familiar format for researchers
   - Best for smaller datasets and standard workflows

2. **Zarr Backend (Compressed Chunks)**:
   - Cutting-edge format with significant compression benefits
   - Requires custom plugins for viewing (napari-zarr, Fiji plugins)
   - Better for large datasets and advanced users
   - OME-ZARR compliance for interoperability

**Additional Optimizations**:
3. **GPU Memory Management**: Keep data on GPU across multiple processing steps to avoid CPU-GPU transfer overhead
4. **Lazy Loading**: Load data only when needed and in the target memory type
5. **Conversion Minimization**: Plan conversion paths to minimize the number of format changes

Real-World Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Scientific Workflow**:
- Input images loaded from disk backend (standard TIFF files)
- All intermediate processing uses memory backend for speed
- Final results materialized to disk or zarr backend based on requirements

**Disk Backend Use Cases**:
- Standard research workflows
- Compatibility with existing tools (ImageJ, Fiji, napari)
- Sharing results with collaborators using standard tools
- Smaller datasets where compression isn't critical

**Zarr Backend Use Cases**:
- Large-scale experiments requiring compression
- Advanced users comfortable with cutting-edge formats
- Workflows requiring OME-ZARR compliance
- Long-term archival with compression benefits

Zarr Chunking Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS provides two chunking strategies for zarr storage, configurable via ``ZarrChunkStrategy`` enum:

**WELL Mode (Default)**:

.. code-block:: python

   zarr_config = ZarrConfig(
       chunk_strategy=ZarrChunkStrategy.WELL,
       compressor=ZarrCompressor.ZSTD,
       compression_level=1
   )

- **Chunk shape**: ``(fields, channels, z, y, x)`` - entire 5D array in one chunk
- **Performance**: 40x improvement for batch operations
- **Use case**: Loading entire wells or many files from same well
- **Example**: 9 fields × 2 channels × 5 z-planes = 1 chunk (~180MB compressed)
- **Benefits**: Optimal for sequential processing, minimal I/O overhead

**FILE Mode**:

.. code-block:: python

   zarr_config = ZarrConfig(
       chunk_strategy=ZarrChunkStrategy.FILE,
       compressor=ZarrCompressor.ZSTD,
       compression_level=1
   )

- **Chunk shape**: ``(1, 1, 1, y, x)`` - each original file is a separate chunk
- **Performance**: Better for random access to individual files
- **Use case**: Sparse access patterns, individual file retrieval
- **Example**: 9 fields × 2 channels × 5 z-planes = 90 chunks (~2MB each)
- **Benefits**: Lower memory footprint, granular access

**Choosing a Strategy**:

- Use **WELL mode** (default) for:
  - Processing entire wells sequentially
  - Batch operations on multiple images
  - Maximum I/O performance
  - Standard high-content screening workflows

- Use **FILE mode** for:
  - Random access to individual images
  - Memory-constrained environments
  - Sparse sampling across wells
  - Interactive exploration workflows

**OME-ZARR Structure**:

Both strategies maintain OME-ZARR HCS compliance with the same 5D array structure:

.. code-block:: text

   /plate_openhcs/images/
   ├── .zgroup                    # Root group metadata
   ├── .zattrs                    # Plate-level OME-ZARR metadata
   ├── A/                         # Row A
   │   ├── 01/                    # Column 01 (well A01)
   │   │   ├── .zgroup
   │   │   └── 0/                 # Field group (5D array)
   │   │       ├── .zarray        # Array metadata
   │   │       ├── .zattrs        # Well metadata + filename mapping
   │   │       └── 0.0.0.0.0      # Chunk file(s)

The difference is internal: WELL mode creates one large chunk file, FILE mode creates many small chunk files.

**Performance Benefits**:
- Automatic format handling between memory types during processing
- Optimized GPU memory management across pipeline steps
- Coordinated storage decisions based on data size and use case
- Consistent performance regardless of memory type used for computation
- Configurable chunking for optimal I/O patterns

Benefits and Design Principles
------------------------------

**System Benefits**:

- **Location Transparency**: Same code works with any storage backend
- **Type Safety**: Automatic conversion with validation prevents silent failures
- **Performance Optimization**: Zero-copy GPU transfers and intelligent conversion strategies
- **Scalability**: Handles datasets from MB to large experimental plates seamlessly
- **Interoperability**: Works with all major scientific Python libraries (NumPy, PyTorch, CuPy, JAX, TensorFlow, pyclesperanto)
- **Fail-Loud Philosophy**: Errors surface immediately rather than corrupting data
- **Automatic Materialization**: Function side effects are automatically converted to appropriate file formats
- **Configuration-Driven Storage**: Backend selection controlled by configuration hierarchy
- **GPU Memory Management**: Automatic cleanup and optimization across all GPU frameworks
- **Thread-Safe Operations**: Thread-local CUDA streams and device management
- **Natural Sorting**: Correct handling of numeric sequences in scientific datasets
- **Batch Operations**: Optimized bulk operations for high-throughput processing

**Design Principles**:

- **Orthogonal Concerns**: Storage, memory type, and materialization decisions are independent
- **Zero-Copy Optimization**: Prioritize GPU-to-GPU transfers using DLPack and CUDA Array Interface
- **Explicit Device Management**: GPU placement is explicit and validated with automatic cleanup
- **Immutable Data Flow**: Data transformations create new objects rather than modifying existing ones
- **Scientific Reproducibility**: All operations are deterministic and traceable
- **Declarative Materialization**: Functions declare their side effects, system handles storage automatically
- **Configuration Hierarchy**: Storage behavior follows the standard OpenHCS configuration resolution
- **Thread-Local Isolation**: Each thread maintains independent GPU contexts and memory management
- **Fail-Fast Validation**: Memory type contracts are enforced at runtime with detailed error reporting
- **Natural Data Ordering**: File operations respect scientific naming conventions and numeric sequences
- **Backend Abstraction**: Unified API across memory, disk, and compressed storage backends

See Also
--------

- :doc:`configuration_framework` - Configuration hierarchy that controls storage behavior
- :doc:`pipeline_compilation_system` - How storage and memory decisions are made during compilation
- :doc:`function_pattern_system` - How functions declare memory type requirements
- :doc:`special_io_system` - Special input/output system that uses materialization
- :doc:`gpu_resource_management` - GPU device management and allocation
- :doc:`memory_type_system` - Memory type system and conversions
- :doc:`../api/index` - API reference (autogenerated from source code)

Archived Documentation
-----------------------

The following documents were consolidated into this unified architecture document:

- ``memory_backend_system.rst`` - VFS backends and storage registry
- ``vfs_system.rst`` - VFS architecture and backend abstraction
- ``memory_type_system.rst`` - Memory type conversion and GPU management

The materialization system content was integrated from existing documentation:

- :doc:`special_io_system` - Cross-step communication patterns using materialization
- Various function implementation files - Real materialization handler examples

Additional content was integrated from:

- ``openhcs/core/memory/decorators.py`` - Memory type declaration system
- ``openhcs/core/memory/conversion_functions.py`` - Memory conversion implementation
- ``openhcs/core/memory/gpu_cleanup.py`` - GPU memory management
- ``openhcs/io/filemanager.py`` - Advanced FileManager operations
- ``openhcs/io/base.py`` - Storage registry and backend architecture

These archived documents are available in ``docs/source/architecture/archive/`` for reference.

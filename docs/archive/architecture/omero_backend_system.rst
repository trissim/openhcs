OMERO Backend System
====================

Overview
--------

The OMERO backend system provides server-side execution support for OpenHCS on OMERO servers with zero data transfer overhead. It implements a virtual backend pattern where filenames are generated on-demand from OMERO metadata rather than from a real filesystem.

**The Server-Side Execution Challenge**: Traditional image processing requires downloading data from OMERO to local machines, processing it, and uploading results back. For high-content screening datasets (100GB+ per plate), this creates massive data transfer bottlenecks and makes server-side processing impractical.

**The OpenHCS Solution**: A virtual backend architecture that generates filenames from OMERO's plate structure without requiring a real filesystem. Combined with multiprocessing-safe connection management and the ZMQ execution system, this enables true server-side processing where data never leaves the OMERO server.

**Key Innovation**: VirtualBackend ABC pattern that separates backends with real filesystems (disk, zarr) from backends that generate filenames on-demand (OMERO, cloud storage). This enables location-transparent processing where the same pipeline code works on disk, zarr, or OMERO without modification.

Architecture
------------

Virtual Backend Pattern
~~~~~~~~~~~~~~~~~~~~~~~

Unlike traditional storage backends (disk, zarr), the OMERO backend is a **VirtualBackend** that generates filenames on-demand from OMERO's plate structure:

.. code-block:: python

   # Traditional backend: Real files on disk
   /data/plate/A01/A01_s001_z000_c000.tif  # Actual file exists

   # OMERO backend: Virtual paths generated from metadata
   /omero/plate_123/A01/A01_s001_z000_c000.tif  # No real file, just metadata

Key Design Principles
~~~~~~~~~~~~~~~~~~~~~

**No Real Filesystem**
  All paths are virtual, generated from OMERO plate structure

**Lazy Loading**
  Plate metadata cached on first access, reused for all operations

**Location Transparency**
  Same path format regardless of backend (disk, zarr, OMERO)

**On-Demand Generation**
  Files created only when needed (e.g., derived plates)

**Automatic Backend Selection**
  OMERO plates automatically use ``omero_local`` for both read and write operations, ignoring user's materialization backend choice

Automatic Backend Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Design Rule**: OMERO plates MUST use ``omero_local`` backend for both input (read) and output (materialization). The system automatically enforces this through the microscope handler's backend compatibility system.

**Why This Matters**:

- OMERO uses virtual paths like ``/omero/plate_123/`` that don't exist on the filesystem
- Attempting to read/write using ``disk`` or ``zarr`` backends will fail with permission errors
- OMERO output must be saved as FileAnnotations attached to OMERO objects, not as files

**Automatic Backend Selection Logic**:

The backend selection happens through the microscope handler's ``compatible_backends`` property:

.. code-block:: python

   class OMEROHandler(MicroscopeHandler):
       @property
       def compatible_backends(self) -> List[Backend]:
           """OMERO is only compatible with OMERO_LOCAL backend."""
           return [Backend.OMERO_LOCAL]

When the compiler calls ``get_primary_backend()``, it returns the first compatible backend, which for OMERO is always ``omero_local``. This applies to both read and materialization backends:

.. code-block:: python

   # Compiler backend selection (in MaterializationFlagPlanner)
   read_backend = context.microscope_handler.get_primary_backend(plate_path, filemanager)
   # Returns: 'omero_local' (first compatible backend for OMERO)

   materialization_backend = context.microscope_handler.get_primary_backend(plate_path, filemanager)
   # Returns: 'omero_local' (same logic)

**User Impact**:

- Users don't need to configure backends for OMERO plates
- System "just works" regardless of VFSConfig settings
- Prevents common errors from trying to write to ``/omero/`` paths

**Contrast with Other Microscopes**:

- **ImageXpress/Opera Phenix**: Compatible with disk backend → Read from disk → Write to OpenHCS format (disk or zarr based on ``materialization_backend``)
- **OpenHCS**: Compatible with disk/zarr/virtual_workspace → Read from auto-detected backend → Write to OpenHCS format (disk or zarr based on ``materialization_backend``)
- **OMERO**: Compatible with omero_local only → Read from omero_local → Write to omero_local (``materialization_backend`` choice ignored)

VirtualBackend ABC
~~~~~~~~~~~~~~~~~~

The system introduces a new abstract base class for backends without real filesystems:

.. code-block:: python

   class VirtualBackend(ABC):
       """Base class for backends without real filesystem (OMERO, cloud, etc.)"""

       @abstractmethod
       def list_files(self, directory: str) -> List[str]:
           """Generate file list from metadata."""
           pass

       @abstractmethod
       def generate_filename(self, metadata: Dict) -> str:
           """Generate filename from metadata."""
           pass

This enables future cloud storage backends (S3, GCS) using the same pattern.

Multiprocessing-Safe Connection Management
------------------------------------------

OMERO connections contain unpicklable ``IcePy.Communicator`` objects, requiring special handling for multiprocessing:

The Problem
~~~~~~~~~~~

.. code-block:: python

   # This fails - connection can't be pickled
   backend = OMEROLocalBackend(omero_conn=conn)
   process = multiprocessing.Process(target=worker, args=(backend,))  # ❌ Pickle error

The Solution
~~~~~~~~~~~~

.. code-block:: python

   # Connection parameters stored, not connection itself
   backend = OMEROLocalBackend(omero_conn=conn)
   # Connection recreated in worker process using stored params
   process = multiprocessing.Process(target=worker, args=(backend,))  # ✅ Works

Implementation Strategy
~~~~~~~~~~~~~~~~~~~~~~~

1. **Main Process**: Store connection parameters (host, port, username, password)
2. **Pickle**: Exclude connection object via ``__getstate__``
3. **Worker Process**: Recreate connection using stored parameters
4. **Global Registry**: Share connections across backend instances

.. code-block:: python

   class OMEROLocalBackend(VirtualBackend):
       def __getstate__(self):
           """Exclude unpicklable connection object."""
           state = self.__dict__.copy()
           # Remove unpicklable connection
           state['_initial_conn'] = None
           return state

       def __setstate__(self, state):
           """Restore state after unpickling."""
           self.__dict__.update(state)
           # Connection will be retrieved from global registry in worker process

See ``openhcs/io/omero_local.py`` lines 93-150 for complete implementation.

Metadata Caching Strategy
--------------------------

OMERO metadata is cached at the plate level to minimize API queries:

Cache Structure
~~~~~~~~~~~~~~~

.. code-block:: python

   @dataclass
   class PlateStructure:
       plate_id: int
       parser_name: str
       microscope_type: str
       wells: Dict[str, WellStructure]  # well_id → WellStructure
       all_well_ids: Set[str]
       max_sites: int
       max_z: int
       max_c: int
       max_t: int

   @dataclass
   class WellStructure:
       well_id: str
       row: int
       col: int
       images: Dict[int, ImageStructure]  # site → ImageStructure

   @dataclass
   class ImageStructure:
       image_id: int
       site: int
       size_z: int
       size_c: int
       size_t: int

Caching Pattern
~~~~~~~~~~~~~~~

.. code-block:: python

   # First access: Query OMERO once for entire plate
   metadata = handler.get_channel_values(plate_id)  # Queries OMERO

   # Subsequent accesses: Return cached data
   z_values = handler.get_z_index_values(plate_id)  # From cache
   t_values = handler.get_timepoint_values(plate_id)  # From cache

This reduces OMERO API calls from O(wells × sites) to O(1) per plate.

Transparent File Handling
--------------------------

Analysis results (JSON/CSV) are automatically saved as OMERO FileAnnotations:

Format Registry
~~~~~~~~~~~~~~~

.. code-block:: python

   class OMEROFileFormatRegistry:
       """Registry of text file formats that should be saved as FileAnnotations."""

       TEXT_FORMATS = {'.json', '.csv', '.txt', '.tsv'}

       @classmethod
       def is_text_format(cls, filename: str) -> bool:
           return Path(filename).suffix.lower() in cls.TEXT_FORMATS

FileAnnotation Creation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def save(self, data: np.ndarray, path: str) -> None:
       """Save data to OMERO (image or FileAnnotation)."""

       if OMEROFileFormatRegistry.is_text_format(path):
           # Save as FileAnnotation
           file_ann = self._create_file_annotation(data, path)
           # Attach to appropriate OMERO object (plate/well/image)
           self._attach_annotation(file_ann, path)
       else:
           # Save as image plane
           self._write_image_plane(data, path)

This is completely transparent to analysis functions - they just call ``filemanager.save()`` and the backend handles OMERO-specific logic.

Automatic Instance Management
------------------------------

The OMERO instance manager provides automatic server lifecycle management:

Auto-Detection
~~~~~~~~~~~~~~

.. code-block:: python

   from openhcs.runtime.omero_instance_manager import OMEROInstanceManager

   manager = OMEROInstanceManager()

   # Auto-detect if OMERO is running
   if manager.is_running():
       print("OMERO is running")
   else:
       print("OMERO is not running")

Auto-Connection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Auto-connect to existing instance
   if manager.connect(timeout=10):
       print(f"Connected to OMERO at {manager.host}:{manager.port}")
       conn = manager.get_connection()
   else:
       print("Failed to connect")

Auto-Start
~~~~~~~~~~

.. code-block:: python

   # Auto-start via docker-compose if not running
   if not manager.is_running():
       manager.start_via_docker_compose()
       manager.wait_for_ready(timeout=60)

Context Manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with OMEROInstanceManager() as manager:
       conn = manager.get_connection()
       # Use connection
       # Automatic cleanup on exit

Integration with ZMQ Execution
-------------------------------

The OMERO backend combines with the ZMQ execution system for server-side processing:

.. code-block:: python

   # Client runs locally
   from openhcs.runtime.zmq_execution_client import ZMQExecutionClient

   client = ZMQExecutionClient(
       host='omero-server.example.com',
       port=7777
   )

   # Server runs on OMERO machine (near data)
   response = client.execute_pipeline(
       plate_id=123,  # OMERO plate ID
       pipeline_steps=steps,
       global_config=config
   )

   # Processing happens server-side
   # Results streamed back to local client
   # Zero data transfer overhead

This pattern eliminates data transfer bottlenecks by processing data where it lives.

Backend Parameter Propagation
------------------------------

All analysis materialization handlers accept a ``backend``/``backends`` parameter to enable saving to any backend:

.. code-block:: python

   def cell_counting_cpu(
       image: np.ndarray,
       filemanager: FileManager,
       metadata: Dict,
       backend: str = 'disk',  # Can be 'disk', 'zarr', or 'omero'
       **params
   ) -> Tuple[np.ndarray, Dict]:
       """Cell counting with backend-agnostic saving."""

       # Process image
       labeled = label_cells(image)

       # Save to specified backend
       filemanager.save(
           labeled,
           construct_path(metadata, 'labeled'),
           backend=backend
       )

       # Save analysis results (JSON)
       results = count_cells(labeled)
       filemanager.save(
           results,
           construct_path(metadata, 'results.json'),
           backend=backend  # Automatically becomes FileAnnotation on OMERO
       )

       return labeled, results

This is completely transparent to analysis code - no OMERO-specific logic needed.

Critical Bug Fix: Black Well Output
------------------------------------

Problem
~~~~~~~

One well in derived plates always had black (zero) output while all others were fine.

Root Cause
~~~~~~~~~~

``_create_derived_plate`` created placeholder images filled with zeros for all wells. When ``_write_planes_to_plate`` ran, it detected these as "already existing" and skipped writing actual data for the first well processed.

Solution
~~~~~~~~

Removed placeholder image creation entirely. Wells are created empty, and images are created with actual data in ``_write_planes_to_plate`` on first write.

.. code-block:: python

   # OLD: Created placeholder zero images
   for well_id, well_data in wells_structure.items():
       well = create_well(plate, well_id)
       for site in range(well_data.max_sites):
           # ❌ Created placeholder with zeros
           image = create_image_with_zeros(well, site)

   # NEW: Create wells without images
   for well_id, well_data in wells_structure.items():
       well = create_well(plate, well_id)
       # ✅ No placeholder images
       # Images created with real data in _write_planes_to_plate

See ``openhcs/io/omero_local.py`` lines 716-730 for implementation.

Usage Example
-------------

Complete Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from polystore.omero_local import OMEROLocalBackend
   from polystore.base import storage_registry
   from openhcs.microscopes.omero import OMEROHandler
   from openhcs.runtime.omero_instance_manager import OMEROInstanceManager
   from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

   # 1. Connect to OMERO
   with OMEROInstanceManager() as manager:
       conn = manager.get_connection()

       # 2. Create and register backend (CRITICAL STEP)
       backend = OMEROLocalBackend(omero_conn=conn)
       storage_registry['omero_local'] = backend

       # 3. Create microscope handler
       handler = OMEROHandler(backend=backend)

       # 4. Run pipeline
       orchestrator = PipelineOrchestrator(
           plate_paths=[123],  # OMERO plate ID
           steps=pipeline_steps,
           global_config=global_config
       )

       orchestrator.run()

       # 5. Results saved as OMERO FileAnnotations
       # Automatically attached to plate/wells/images

**Critical Note**: The OMERO backend must be manually registered in the ``storage_registry`` because it requires a connection object that cannot be created automatically by the metaclass system. This is different from other backends (disk, memory, zarr) which are auto-registered.

See Also
--------

- :doc:`storage_and_memory_system` - Storage backend architecture
- :doc:`../guides/omero_integration` - OMERO integration guide

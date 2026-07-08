# OMERO Integration for OpenHCS

Complete OMERO server-side execution support for OpenHCS, enabling high-performance image processing directly on OMERO servers with zero data transfer overhead.

---

## 🚀 Quick Start (Testing)

**TL;DR:** Start OMERO, run the test.

```bash
# 1. Start OMERO
cd openhcs/omero
docker-compose up -d
./wait_for_omero.sh

# 2. Run integration tests
cd ../..
pytest tests/integration/test_main.py --it-microscopes=OMERO --it-backends=disk -v
```

**See:** [`OMERO_TESTING_GUIDE.md`](../../OMERO_TESTING_GUIDE.md) in project root for complete testing documentation.

---

## Overview

The OMERO integration provides:

- **Virtual Backend**: On-demand filename generation from OMERO plate structure (no real filesystem)
- **Native Metadata**: Direct OMERO API queries with per-plate caching
- **Multiprocessing-Safe**: Connection management that works across process boundaries
- **Automatic Instance Management**: Auto-detect, connect, and start OMERO via Docker
- **Transparent File Handling**: JSON/CSV analysis results saved as OMERO FileAnnotations
- **Zero-Copy Access**: Server-side execution eliminates data transfer bottlenecks

## Architecture

### Virtual Backend Pattern

Unlike traditional storage backends (disk, zarr), the OMERO backend is a **VirtualBackend** that generates filenames on-demand from OMERO's plate structure:

```python
# Traditional backend: Real files on disk
/data/plate/A01/A01_s001_z000_c000.tif  # Actual file

# OMERO backend: Virtual paths generated from metadata
/omero/plate_123/A01/A01_s001_z000_c000.tif  # No real file, just metadata
```

**Key Design Principles:**

1. **No Real Filesystem**: All paths are virtual, generated from OMERO plate structure
2. **Lazy Loading**: Plate metadata cached on first access, reused for all operations
3. **Location Transparency**: Same path format regardless of backend
4. **On-Demand Generation**: Files created only when needed (e.g., derived plates)

### Multiprocessing-Safe Connection Management

OMERO connections contain unpicklable `IcePy.Communicator` objects, requiring special handling for multiprocessing:

**Problem:**
```python
# This fails - connection can't be pickled
backend = OMEROLocalBackend(omero_conn=conn)
process = multiprocessing.Process(target=worker, args=(backend,))  # ❌ Pickle error
```

**Solution:**
```python
# Connection parameters stored, not connection itself
backend = OMEROLocalBackend(omero_conn=conn)
# Connection recreated in worker process using stored params
process = multiprocessing.Process(target=worker, args=(backend,))  # ✅ Works
```

**Implementation:**

1. **Main Process**: Store connection parameters (host, port, username)
2. **Pickle**: Exclude connection object via `__getstate__`
3. **Worker Process**: Recreate connection using stored parameters
4. **Global Registry**: Share connections across backend instances

See `openhcs/io/omero_local.py` lines 93-150 for implementation details.

### Metadata Caching Strategy

OMERO metadata is cached at the plate level to minimize API queries:

```python
# First access: Query OMERO once for entire plate
metadata = handler.get_channel_values(plate_id)  # Queries OMERO
# Subsequent accesses: Return cached data
z_values = handler.get_z_index_values(plate_id)  # From cache
t_values = handler.get_timepoint_values(plate_id)  # From cache
```

**Cache Structure:**
```python
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
```

## Quick Start

### 1. Start OMERO Server

```bash
# Using Docker Compose (automatic via instance manager)
cd openhcs/omero
docker-compose up -d

# Wait for OMERO to be ready
./wait_for_omero.sh
```

### 2. Run Integration Tests

```bash
# Run OMERO integration tests (generates and uploads test data automatically)
pytest tests/integration/test_main.py --it-microscopes=OMERO --it-backends=disk -v

# Tests automatically:
# - Generate synthetic microscopy data
# - Upload to OMERO as a Plate
# - Run full OpenHCS pipeline
# - Save results back to OMERO as FileAnnotations
```

### 3. Run Pipeline on OMERO Plate

```python
from openhcs.omero import OMEROInstanceManager, OMEROLocalBackend
from polystore.base import storage_registry
from openhcs.core.orchestrator import Orchestrator
from openhcs.core.config import PipelineConfig

# Connect to OMERO
with OMEROInstanceManager() as manager:
    # Register OMERO backend
    backend = OMEROLocalBackend(omero_conn=manager.conn)
    storage_registry['omero_local'] = backend
    
    # Create pipeline definition
    pipeline_steps = [...]
    pipeline_config = PipelineConfig()
    
    # Execute on OMERO plate (use plate_id as path)
    orchestrator = Orchestrator(
        plate_dir=f"/omero/plate_{plate_id}",
        pipeline_steps=pipeline_steps,
        pipeline_config=pipeline_config,
        backend='omero_local'
    )
    orchestrator.run()
```

### 4. View Results

Results are automatically opened in your browser at:
```
http://localhost:4080/webclient/?show=plate-{plate_id}
```

## Core Components

### OMEROLocalBackend

**Location:** `openhcs/io/omero_local.py`

Virtual backend implementing the VirtualBackend interface:

```python
from openhcs.omero import OMEROLocalBackend

backend = OMEROLocalBackend(omero_conn=conn)

# Virtual filesystem operations
files = backend.list_files("/omero/plate_123/A01")  # Generated from metadata
data = backend.load("/omero/plate_123/A01/A01_s001.tif")  # Loads from OMERO
backend.save(data, "/omero/plate_456/A01/processed.tif")  # Creates new image
```

**Key Methods:**
- `list_files()`: Generate file list from plate structure
- `load()`: Load image data from OMERO
- `save()`: Create new OMERO images or FileAnnotations
- `save_batch()`: Batch create images with well grouping

### OMEROMicroscopeHandler

**Location:** `openhcs/microscopes/omero.py`

Microscope handler for OMERO native metadata:

```python
from openhcs.omero import OMEROMicroscopeHandler

handler = OMEROMicroscopeHandler(filemanager)

# Query OMERO metadata (cached per plate)
channels = handler.metadata_handler.get_channel_values(plate_id)
z_planes = handler.metadata_handler.get_z_index_values(plate_id)
timepoints = handler.metadata_handler.get_timepoint_values(plate_id)
```

**Features:**
- Native OMERO metadata queries
- Per-plate caching
- Dynamic metadata getter generation
- Connection retrieval from backend registry

### OMEROInstanceManager

**Location:** `openhcs/runtime/omero_instance_manager.py`

Automatic OMERO instance management:

```python
from openhcs.omero import OMEROInstanceManager

# Context manager (recommended)
with OMEROInstanceManager() as manager:
    conn = manager.conn
    # Use connection...
# Connection automatically closed

# Manual management
manager = OMEROInstanceManager()
if manager.connect(timeout=60):
    # Connected to existing or started new instance
    conn = manager.conn
    # Use connection...
    manager.close()
```

**Features:**
- Auto-detect if OMERO is running
- Auto-connect to existing instance
- Auto-start via docker-compose if not running
- Context manager support
- Configurable timeouts

## Testing

### Running OMERO Tests

OMERO tests are integrated into the main test suite via `test_main.py`:

```bash
# Run OMERO tests (auto-starts OMERO if needed)
pytest tests/integration/test_main.py -k omero -v

# Run specific OMERO test
pytest tests/integration/test_main.py::test_main[omero_plate] -v
```

**Test Pattern:**

The test suite uses `plate_dir: int` (plate_id) to trigger OMERO mode:

```python
# Disk-based test
test_main(plate_dir=Path("/data/plate"), backend="disk", ...)

# OMERO test (plate_id instead of Path)
test_main(plate_dir=123, backend="omero_local", ...)
```

When `plate_dir` is an integer:
1. Auto-detect and connect to OMERO
2. Register OMERO backend with connection
3. Convert plate_id to virtual path: `/omero/plate_123`
4. Execute pipeline
5. Auto-open browser to view results

### Test Automation

The instance manager provides full automation:

```python
# In test_main.py
if test_config.is_omero:
    omero_manager = OMEROInstanceManager()
    if not omero_manager.connect(timeout=60):
        pytest.skip("OMERO not available")
    # OMERO is now running and connected
```

**Automation Features:**
- ✅ Auto-detect OMERO running state
- ✅ Auto-connect to existing instance
- ✅ Auto-start via docker-compose if not running
- ✅ Auto-open browser to view results
- ✅ Configurable timeouts (60s startup, 10s connection)

## Configuration

### Docker Compose

**Location:** `openhcs/omero/docker-compose.yml`

Default configuration:
- **OMERO Server**: Port 4064
- **OMERO Web**: Port 4080
- **PostgreSQL**: Port 5432 (internal)
- **Root Password**: `openhcs`
- **Database**: `omero` / `omero` / `omero`

### Environment Variables

```bash
# OMERO connection
export OMERO_HOST=localhost
export OMERO_PORT=4064
export OMERO_USER=root
export OMERO_PASSWORD=openhcs

# OMERO Web
export OMERO_WEB_HOST=localhost
export OMERO_WEB_PORT=4080
```

### Backend Registration

```python
from openhcs.omero import OMEROLocalBackend
from polystore.base import storage_registry

# Register OMERO backend globally
backend = OMEROLocalBackend(omero_conn=conn)
storage_registry['omero_local'] = backend

# Now available to all components
from openhcs.constants.constants import Backend
filemanager = FileManager(backend=Backend.OMERO_LOCAL)
```

## Advanced Usage

### Custom Plate Creation

```python
# Create derived plate with custom metadata
plate_id = backend._create_derived_plate(
    plate_name="Processed_Plate",
    wells_structure={
        'A01': {
            'sites': {
                0: {
                    'sizeZ': 5,
                    'sizeC': 2,
                    'sizeT': 1,
                    'height': 512,
                    'width': 512,
                    'dtype': np.uint16
                }
            }
        }
    },
    parser_name='OMEROFilenameParser',
    microscope_type='omero'
)
```

### FileAnnotation Handling

JSON/CSV files are automatically saved as OMERO FileAnnotations:

```python
# Save analysis results (automatically becomes FileAnnotation)
backend.save(json_content, "/omero/plate_123/analysis_results.json", "omero_local")

# OMERO backend detects .json extension and creates FileAnnotation
# Attached to appropriate OMERO object (plate/well/image)
```

### Connection Sharing

Multiple backend instances can share the same connection:

```python
# Main process
backend1 = OMEROLocalBackend(omero_conn=conn)
storage_registry['omero_local'] = backend1

# Worker process (connection retrieved from registry)
backend2 = OMEROLocalBackend()  # No connection passed
conn = backend2._get_connection()  # Retrieved from registry
```

## Troubleshooting

### OMERO Won't Start

```bash
# Check Docker status
docker-compose -f openhcs/omero/docker-compose.yml ps

# View logs
docker-compose -f openhcs/omero/docker-compose.yml logs

# Restart OMERO
docker-compose -f openhcs/omero/docker-compose.yml restart
```

### Connection Timeout

```python
# Increase timeout
manager = OMEROInstanceManager()
manager.connect(timeout=120)  # 2 minutes
```

### Pickle Errors in Multiprocessing

If you see `IcePy.Communicator` pickle errors:

1. Ensure connection is passed via kwargs, not stored in instance
2. Use global registry pattern for connection sharing
3. Check `__getstate__` excludes connection object

### Black Well Bug

**Fixed in this release!** If you see one well with black output:

- **Cause**: Placeholder zero images created before actual data
- **Fix**: Removed placeholder creation, images created with real data
- **Location**: `openhcs/io/omero_local.py` lines 716-730

## Performance Considerations

### Zero-Copy Access

Server-side execution eliminates data transfer:

```
Traditional: OMERO → Download → Process → Upload → OMERO
OMERO Backend: OMERO → Process (in-place) → OMERO
```

**Speedup:** 10-100x for large plates (no network transfer)

### Metadata Caching

Plate metadata cached on first access:

```
First well: 1 OMERO query (entire plate)
Subsequent wells: 0 OMERO queries (cached)
```

**Speedup:** 100x for multi-well plates

### Batch Operations

Images created in batches per well:

```python
# Efficient: One transaction per well
backend.save_batch(images, paths, 'omero_local')

# Inefficient: One transaction per image
for img, path in zip(images, paths):
    backend.save(img, path, 'omero_local')
```

## See Also

- **Setup Guide**: `openhcs/omero/SETUP.md`
- **Docker Compose**: `openhcs/omero/docker-compose.yml`
- **Backend Implementation**: `openhcs/io/omero_local.py`
- **Microscope Handler**: `openhcs/microscopes/omero.py`
- **Instance Manager**: `openhcs/runtime/omero_instance_manager.py`
- **Integration Tests**: `tests/integration/test_main.py`

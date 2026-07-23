Napari Integration Architecture
================================

Overview
--------

OpenHCS implements a sophisticated real-time streaming architecture for Napari visualization that enables live monitoring of high-content screening image processing pipelines. The integration uses inter-process communication (IPC) via ZeroMQ and shared memory to stream processed images from OpenHCS worker processes to persistent Napari viewer instances with minimal latency and zero data copying overhead.

**The Visualization Challenge**: Traditional visualization approaches embed viewers in the main process, causing Qt threading conflicts and blocking pipeline execution. For high-content screening with hundreds of images, this creates an impossible choice between visualization and performance.

**The OpenHCS Solution**: A process-based streaming architecture that separates visualization into independent processes communicating via ZeroMQ. This eliminates Qt threading issues while enabling true real-time monitoring without performance impact on pipeline execution.

**Key Innovation**: Zero-copy shared memory transfer combined with component-aware layer organization automatically groups images by microscopy metadata (wells, channels, z-planes), providing intuitive navigation of complex datasets.

Core Design Principles
----------------------

Zero-Copy Data Transfer
~~~~~~~~~~~~~~~~~~~~~~~

Uses shared memory blocks for efficient transfer of large image arrays between processes without serialization overhead.

.. code-block:: python

   # Pipeline Worker Process
   GPU/CPU Array → NumPy conversion
       ↓
   Shared Memory Block creation
       ↓
   ZeroMQ JSON message (metadata + shm_name)
       ↓
   Napari Viewer Process

Persistent Viewer Processes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Napari viewers survive pipeline completion, enabling examination of intermediate results across multiple pipeline runs. Viewers can be reused by different processes (pipelines, image browser, tests).

Component-Aware Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically organizes images into layers based on microscopy metadata (wells, channels, z-planes, sites, timepoints). No manual layer management required.

Dual-Channel Communication
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separates data streaming (PUB/SUB) from control messages (REQ/REP) for reliable handshaking and process management:

- **Data Port** (e.g., 5555): PUB/SUB socket for streaming progress updates
- **Control Port** (data_port + 1000, e.g., 6555): REQ/REP socket for ping/pong handshake

Multi-Instance Support
~~~~~~~~~~~~~~~~~~~~~~~

Manages multiple concurrent Napari viewers on different ports, each with independent configuration. Multiple researchers can monitor different aspects of the same pipeline.

Architecture Components
-----------------------

Streaming Backend
~~~~~~~~~~~~~~~~~

**Location**: ``openhcs/io/napari_stream.py``

The Napari streaming backend integrates into OpenHCS's unified backend system alongside disk, memory, and zarr backends. It implements the ``StreamingBackend`` interface.

**Key Features**:

- Automatic registration via metaclass system
- Batch streaming support for efficient multi-image transfer
- Shared memory management with automatic cleanup
- ZeroMQ publisher connection pooling per port
- GPU-to-CPU tensor conversion (PyTorch, CuPy, JAX)

.. code-block:: python

   # Streaming backend usage
   filemanager.save_batch(
       data=images,
       paths=paths,
       backend='napari_stream'
   )

Viewer Process Manager
~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``openhcs/runtime/napari_stream_visualizer.py``

Manages the lifecycle of Napari viewer processes, including spawning, connection management, and graceful shutdown.

**Process Management**:

- **Persistent Mode**: Spawns detached subprocess using ``os.setsid()`` that survives parent termination
- **Non-Persistent Mode**: Uses ``multiprocessing.Process`` for test scenarios requiring cleanup
- **Connection Reuse**: Attempts to connect to existing viewers on the same port before spawning new processes
- **Handshake Protocol**: Ping/pong verification ensures viewer responsiveness before streaming

**Port Management**:

- Data port (default: 5555): ZeroMQ PUB/SUB for image streaming
- Control port (data_port + 1000): ZeroMQ REQ/REP for handshake and control messages
- Automatic port conflict detection and resolution
- Process-based port killing for unresponsive viewers

.. code-block:: python

   # Start persistent viewer
   visualizer = NapariStreamVisualizer(
       filemanager,
       visualizer_config,  # Configuration for streaming behavior
       viewer_title="OpenHCS Pipeline Visualization",
       persistent=True,
       napari_port=5555,
       replace_layers=False
   )
   visualizer.start_viewer(async_mode=True)  # Non-blocking startup

Viewer Process
~~~~~~~~~~~~~~

**Location**: ``openhcs/runtime/napari_stream_visualizer.py::_napari_viewer_process``

The actual Napari viewer runs in a separate Python process with its own Qt event loop, receiving and displaying images via ZeroMQ.

**Message Processing**:

- Qt timer-based polling (50ms intervals) for responsive UI
- Batch message handling (up to 10 messages per tick)
- Shared memory attachment and array reconstruction
- Component-aware layer organization
- Automatic cleanup of processed shared memory blocks

**Layer Management**:

- Separate layers per processing step and well
- Component-based stacking (channels as slices, z-planes as stacks)
- Configurable dimension handling (SLICE vs STACK modes)
- Variable size handling (separate layers or padding to max size)
- Dynamic layer updates vs. creation based on replace_layers flag

Component-Aware Display System
-------------------------------

OpenHCS automatically parses microscopy filenames to extract component metadata (well, site, channel, z-index, timepoint) and uses this information to intelligently organize images in Napari.

Stacking Logic
~~~~~~~~~~~~~~

**SLICE Mode**
  Creates separate 2D layers for each component value (e.g., separate layers per channel)

**STACK Mode**
  Combines component values into 3D volumes (e.g., z-planes stacked into volumes)

**Default Behavior**
  Channels as slices, all other dimensions as stacks

Layer Naming Convention
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Step_{step_index}_{step_name}_Well_{well_id}_[Component_Values]

Example: ``Step_02_gaussian_blur_Well_A01_Channel_DAPI_Site_001``

Variable Size Handling
~~~~~~~~~~~~~~~~~~~~~~

When images within the same layer have different dimensions (common in multi-well plates with varying field sizes):

**SEPARATE_LAYERS**
  Creates individual layers per well (preserves exact data)

**PAD_TO_MAX**
  Pads smaller images to match the largest (enables stacking)

Configuration System
--------------------

The Napari integration uses OpenHCS's lazy configuration framework with placeholder inheritance:

.. code-block:: python

   @dataclass(frozen=True)
   class NapariStreamingConfig(StreamingConfig, NapariDisplayConfig):
       napari_port: int = 5555
       napari_host: str = 'localhost'

       # Inherited from NapariDisplayConfig:
       colormap: NapariColormap = NapariColormap.GRAY
       variable_size_handling: NapariVariableSizeHandling = SEPARATE_LAYERS
       site_mode: NapariDimensionMode = STACK
       channel_mode: NapariDimensionMode = SLICE
       z_index_mode: NapariDimensionMode = STACK
       well_mode: NapariDimensionMode = STACK

**Dynamic Configuration**:

- Colormap enum auto-generated from Napari's available colormaps via introspection
- Component mode fields dynamically created based on OpenHCS component configuration
- Configuration inherits through pipeline → step → function parameter hierarchy

Integration Points
------------------

Pipeline Execution
~~~~~~~~~~~~~~~~~~

During pipeline execution, the streaming backend is activated alongside persistent backends:

.. code-block:: python

   # Pipeline configuration enables Napari streaming
   pipeline_config = PipelineConfig(
       napari_streaming=NapariStreamingConfig(
           enabled=True,
           persistent=True,
           napari_port=5555
       )
   )

The visualizer starts asynchronously in a background thread, allowing pipeline execution to proceed without waiting for viewer initialization.

Desktop GUI Integration
~~~~~~~~~~~~~~~~~~~~~~~

The PyQt6 GUI provides a Napari instance manager with:

- **Instance List**: Shows all running Napari viewers with port numbers and status
- **Launch Controls**: Start new viewers with custom ports and titles
- **Image Streaming**: Stream selected images from the image browser to any viewer
- **Process Management**: Graceful quit (with pong verification) or force kill (port-based)
- **Queue Management**: Buffers images when viewer is starting up

**Viewer Status Tracking**:

- Green indicator: Viewer running and responsive
- Red indicator: Viewer not responding or terminated
- Queue count: Number of pending images waiting for viewer startup

Remote Execution Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The streaming architecture naturally extends to remote execution scenarios:

**Local Execution**:

.. code-block:: text

   OpenHCS Process → localhost:5555 → Napari Process (same machine)

**Remote Execution**:

.. code-block:: text

   OpenHCS Server (remote) → network:5555 → Napari Viewer (local workstation)

The ``napari_host`` configuration parameter controls the target:

- ``localhost``: Local IPC streaming
- Remote IP address: Network streaming to remote viewer

This enables the preferred workflow where OpenHCS runs server-side near large datasets and streams results back to the researcher's local workstation.

Performance Characteristics
---------------------------

Memory Efficiency
~~~~~~~~~~~~~~~~~

- **Zero-Copy Transfer**: Shared memory avoids data duplication between processes
- **Automatic Cleanup**: Shared memory blocks released after viewer processes them
- **Batch Streaming**: Multiple images sent in single ZeroMQ message reduces overhead

Latency
~~~~~~~

- **Asynchronous Startup**: Viewer initialization doesn't block pipeline execution
- **High-Frequency Polling**: 50ms timer intervals provide responsive updates
- **Batch Processing**: Up to 10 messages processed per timer tick for high throughput

Scalability
~~~~~~~~~~~

- **Multi-Instance Support**: Multiple viewers can run simultaneously on different ports
- **Process Isolation**: Each viewer runs in separate process with independent Qt event loop
- **Connection Pooling**: ZeroMQ publishers reused across pipeline steps

Use Cases
---------

Pipeline Development and Debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Researchers can monitor processing results in real-time as the pipeline executes, enabling rapid iteration on analysis parameters without waiting for full pipeline completion.

Quality Control
~~~~~~~~~~~~~~~

Live visualization allows immediate detection of processing artifacts, segmentation failures, or parameter misconfigurations during long-running batch analyses.

Multi-Step Workflow Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Persistent viewers retain all intermediate results, allowing researchers to compare outputs from different processing steps side-by-side.

Remote High-Performance Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Researchers can run computationally intensive pipelines on remote GPU servers while visualizing results locally, avoiding the need to download large datasets.

Multi-Well Plate Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Component-aware layer organization automatically groups images by well and channel, providing intuitive navigation of high-content screening datasets with hundreds of fields of view.

See Also
--------

- :doc:`napari_streaming_system` - Materialization-aware streaming
- :doc:`../guides/viewer_management` - Viewer management guide
- :doc:`../user_guide/real_time_visualization` - Real-time visualization guide

Fiji Streaming System
=====================

Overview
--------

Pipeline visualization with Fiji/ImageJ requires real-time data streaming to external processes without blocking pipeline execution. The Fiji streaming system provides automatic hyperstack creation and PyImageJ integration for leveraging the ImageJ/Fiji ecosystem while maintaining OpenHCS's performance characteristics.

**The Fiji Integration Challenge**: ImageJ/Fiji uses a different dimensional model (CZT: Channels, Z-slices, Time-frames) than OpenHCS's component-based system. Additionally, hyperstack building is computationally expensive (~2 seconds per stack), which could block pipeline execution if not handled properly.

**The OpenHCS Solution**: A thin-wrapper architecture where OpenHCS composes
external streaming infrastructure. Generic transport is provided by
``zmqruntime``, payload semantics and receiver-core utilities are provided by
``polystore``, and OpenHCS provides runtime wiring and process management.

Canonical Abstraction Docs
--------------------------

For generic abstraction internals, see external module docs:

- ``external/zmqruntime/docs/source/architecture/viewer_streaming_architecture.rst``
- ``external/PolyStore/docs/source/architecture/streaming_receiver_projection.rst``

This page focuses on OpenHCS-specific viewer wrapper behavior.

**Key Innovation**: Shared memory IPC with proper lifecycle management ensures zero-copy data transfer while preventing memory leaks. The publisher closes handles after successful sends, while the receiver unlinks shared memory after copying data.

Architecture Components
-----------------------

FijiStreamingBackend
~~~~~~~~~~~~~~~~~~~~

**Location**: ``external/PolyStore/src/polystore/fiji_stream.py``

The streaming backend sends image data to Fiji viewers using ZeroMQ publish/subscribe pattern:

.. code-block:: python

   class FijiStreamingBackend(StreamingBackend):
       """Fiji streaming backend with ZMQ publisher pattern."""

       def save_batch(self, data_list, file_paths, **kwargs):
           # Create shared memory for zero-copy transfer
           shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
           shm_array = np.ndarray(shape, dtype, buffer=shm.buf)
           shm_array[:] = data[:]

           # Send metadata + shared memory name (not data)
           message = {
               'images': [{
                   'shm_name': shm.name,
                   'shape': data.shape,
                   'dtype': str(data.dtype),
                   'component_metadata': metadata
               }],
               'display_config': config
           }
           publisher.send_json(message, flags=zmq.NOBLOCK)

           # Clean up publisher's handle after successful send
           shm.close()  # Receiver will unlink after copying

**Key Features**:

- **Zero-copy transfer**: Shared memory IPC for large image arrays
- **Non-blocking sends**: ``zmq.NOBLOCK`` flag prevents pipeline blocking
- **High water mark**: Increased to 10000 to buffer slow hyperstack building
- **Proper cleanup**: Publisher closes handles after send, receiver unlinks after copy

FijiViewerServer
~~~~~~~~~~~~~~~~

**Location**: ``openhcs/runtime/fiji_viewer_server.py`` (OpenHCS wrapper)

The viewer server receives images and displays them via PyImageJ, inheriting from ``ZMQServer`` ABC:

.. code-block:: python

   class FijiViewerServer(ZMQServer):
       """ZMQ server for Fiji viewer with PyImageJ integration."""

       def start(self):
           # Initialize PyImageJ in this process
           import imagej
           self.ij = imagej.init(mode='interactive')
           self.ij.ui().showUI()

       def process_image_message(self, message):
           # Attach to shared memory
           shm = shared_memory.SharedMemory(name=shm_name)
           data = np.ndarray(shape, dtype, buffer=shm.buf).copy()
           shm.close()
           shm.unlink()  # Clean up shared memory

           # Build hyperstack from component metadata
           hyperstack = self._build_hyperstack(images, metadata)
           self.ij.ui().show(hyperstack)

**Key Features**:

- **PyImageJ integration**: Uses ``imagej.init()`` for native Fiji functionality
- **Hyperstack building**: Automatic CZT dimension mapping from component metadata
- **Dual-channel ZMQ**: Control channel (REQ/REP) + data channel (PUB/SUB)
- **Ping/pong handshake**: Inherited from ``ZMQServer`` ABC for connection verification

FijiStreamVisualizer
~~~~~~~~~~~~~~~~~~~~

**Location**: ``openhcs/runtime/fiji_stream_visualizer.py`` (OpenHCS wrapper)

Manages Fiji viewer process lifecycle, following the same architecture as ``NapariStreamVisualizer``:

.. code-block:: python

   class FijiStreamVisualizer:
       """Manages Fiji viewer instance for real-time visualization."""

       def start_viewer(self, async_mode: bool = False):
           # Check for existing viewer on same port
           if self._try_connect_to_existing_viewer():
               return

           # Spawn new viewer process
           if self.persistent:
               # Detached subprocess survives parent termination
               self.process = _spawn_detached_fiji_process(port, title, config)
           else:
               # Multiprocessing.Process for test cleanup
               self.process = multiprocessing.Process(
                   target=_fiji_viewer_server_process,
                   args=(port, title, config, log_file)
               )
               self.process.start()

           # Wait for ping/pong handshake
           if not async_mode:
               self._wait_for_server_ready()

**Key Features**:

- **Persistent viewers**: Detached subprocesses survive parent termination
- **Viewer reuse**: Connects to existing viewers before spawning new ones
- **Async startup**: Background thread startup for non-blocking initialization
- **Process management**: Graceful shutdown with timeout and force-kill fallback

Shared Memory Lifecycle
------------------------

The shared memory lifecycle is critical for preventing memory leaks while maintaining zero-copy performance:

.. code-block:: text

   Publisher (FijiStreamingBackend):
   1. Create shared memory block
   2. Copy data into shared memory
   3. Send message with shared memory name
   4. Close handle (but don't unlink)
   5. Remove from tracking dict

   Receiver (FijiViewerServer):
   1. Receive message with shared memory name
   2. Attach to shared memory
   3. Copy data from shared memory
   4. Close handle
   5. Unlink shared memory (cleanup)

**Why This Works**: The publisher closes its handle immediately after sending, preventing handle accumulation. The receiver unlinks the shared memory after copying, ensuring cleanup even if the publisher crashes.

**Previous Bug**: The publisher never closed handles after successful sends, causing the ``_shared_memory_blocks`` dict to grow indefinitely with stale references. This was fixed in commit ``6bd32f1``.

Hyperstack Building
-------------------

Component to CZT Mapping
~~~~~~~~~~~~~~~~~~~~~~~~

OpenHCS uses component-based dimensions (well, site, channel, z_index, timepoint), while Fiji uses CZT (Channels, Z-slices, Time-frames). The system automatically maps between these:

.. code-block:: python

   # Component metadata from OpenHCS
   metadata = {
       'well': 'A01',
       'site': 's1',
       'channel': 'DAPI',
       'z_index': 'z003',
       'timepoint': 't001'
   }

   # Automatic CZT mapping based on FijiDisplayConfig
   config = FijiDisplayConfig(
       channel_mode=FijiDimensionMode.SLICE,  # C dimension
       z_index_mode=FijiDimensionMode.STACK,  # Z dimension
       timepoint_mode=FijiDimensionMode.STACK # T dimension
   )

   # Result: Hyperstack with nChannels=1, nSlices=N, nFrames=M

**Dimension Modes**:

- ``STACK``: Component becomes a hyperstack dimension (C, Z, or T)
- ``SLICE``: Component values shown as separate slices
- ``SEPARATE``: Component values shown as separate images

Hyperstack Creation
~~~~~~~~~~~~~~~~~~~

The server groups images by (step_name, well) and builds hyperstacks:

.. code-block:: python

   def _build_hyperstack(self, images, display_config):
       # Group images by CZT coordinates
       czt_map = {}
       for img in images:
           c = self._get_channel_index(img['metadata'])
           z = self._get_z_index(img['metadata'])
           t = self._get_timepoint_index(img['metadata'])
           czt_map[(c, z, t)] = img['data']

       # Create ImageJ ImageStack
       stack = self.ij.py.to_java(first_image)
       for (c, z, t), data in sorted(czt_map.items()):
           stack.addSlice(self.ij.py.to_java(data))

       # Convert to hyperstack
       imp = ImagePlus(title, stack)
       imp.setDimensions(nChannels, nSlices, nFrames)

       # Apply LUT and display mode
       if nChannels > 1:
           comp = CompositeImage(imp, CompositeImage.COMPOSITE)
           return comp
       return imp

**Performance Note**: Hyperstack building takes ~2 seconds per stack due to ImageJ's internal processing. The non-blocking send pattern ensures this doesn't block the pipeline.

ZeroMQ Communication Pattern
----------------------------

Dual-Channel Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~

Fiji streaming uses the same dual-channel pattern as Napari and OMERO integrations:

.. code-block:: text

   Control Channel (REQ/REP):
   - Port: fiji_port + 1000 (e.g., 6556 for data port 5556)
   - Purpose: Ping/pong handshake, status queries
   - Pattern: Synchronous request/response

   Data Channel (PUB/SUB):
   - Port: fiji_port (e.g., 5556)
   - Purpose: Image streaming
   - Pattern: Asynchronous publish/subscribe

**Benefits**:

- Control messages don't block data streaming
- Reliable handshake before data transfer
- Independent scaling of control and data throughput

Non-Blocking Sends
~~~~~~~~~~~~~~~~~~

The publisher uses ``zmq.NOBLOCK`` to prevent pipeline blocking:

.. code-block:: python

   try:
       publisher.send_json(message, flags=zmq.NOBLOCK)
       # Clean up shared memory handles after successful send
       for img in batch_images:
           shm = self._shared_memory_blocks.pop(img['shm_name'])
           shm.close()
   except zmq.Again:
       # Fiji viewer busy, drop batch and clean up
       logger.warning(f"Fiji viewer busy, dropped batch")
       for img in batch_images:
           shm = self._shared_memory_blocks.pop(img['shm_name'])
           shm.close()
           shm.unlink()  # Unlink since receiver never got it

**Why This Works**: If the receiver's buffer is full (``zmq.Again`` exception), the publisher drops the batch and cleans up shared memory. This prevents pipeline blocking while ensuring no memory leaks.

Configuration System
--------------------

The Fiji integration uses OpenHCS's lazy configuration framework with placeholder inheritance:

.. code-block:: python

   @dataclass(frozen=True)
   class FijiStreamingConfig(StreamingConfig, FijiDisplayConfig):
       fiji_port: int = 5556
       fiji_host: str = 'localhost'

       # Inherited from FijiDisplayConfig:
       lut: FijiLUT = FijiLUT.GRAYS
       channel_mode: FijiDimensionMode = SLICE
       z_index_mode: FijiDimensionMode = STACK
       timepoint_mode: FijiDimensionMode = STACK
       site_mode: FijiDimensionMode = SEPARATE

**Lazy Configuration**: Steps can use ``LazyFijiStreamingConfig()`` to inherit from pipeline-level defaults, enabling centralized configuration with per-step overrides.

See Also
--------

- :doc:`external_integrations_overview` - Overview of all OpenHCS integrations
- :doc:`napari_streaming_system` - Napari streaming (similar architecture)
- :doc:`../guides/fiji_viewer_management` - Fiji viewer management guide

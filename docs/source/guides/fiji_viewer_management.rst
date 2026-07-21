Fiji streaming
==============

Fiji streaming uses the same compiled streaming boundary as Napari, with an
OpenHCS Fiji adapter and a detached Fiji/ImageJ viewer process.

Setup
-----

Install ``openhcs[gui,fiji]`` or ``openhcs[gui,viz]`` and verify that the
PyImageJ/Java environment can start independently. In the Pipeline Editor,
enable the Fiji streaming configuration for the step and compile before
running.

OpenHCS compiles component order and display metadata into the stream plan. The
viewer adapter constructs ImageJ-compatible displays from the typed payload; a
pipeline declaration does not launch Fiji directly and should not embed a ZMQ
port or hand-built wire message.

Lifecycle
---------

ZMQRuntime owns generic readiness checks, control messages, acknowledgments, and
process management. OpenHCS owns the Fiji entry point, ImageJ conversion, macro
integration, and viewer-specific presentation. Persistent viewers may be reused
only after the control protocol confirms that the bound process is compatible.

For shared behavior, see :doc:`viewer_management` and
:doc:`../architecture/streaming_boundary_and_wrappers`.

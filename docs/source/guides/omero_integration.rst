OMERO integration
=================

OMERO support crosses three ownership boundaries:

``omero_openhcs``
  Owns the OMERO.web application, templates, and application-level integration.
  Its source is bundled inside the OpenHCS distribution at
  ``openhcs/omero/plugin`` rather than installed from a repository-root project.

PolyStore
  Owns generic storage backends, virtual paths, source references, and ROI
  persistence primitives.

OpenHCS
  Owns microscope/source selection, source bindings, compilation, processing,
  the desktop workflows that choose an OMERO source, and the packaged
  ``openhcs/omero`` deployment bundle and instance lifecycle.

The current PolyStore ``OMEROLocalBackend`` still imports the OpenHCS
``FilenameParser`` registry while building its virtual source projection. This
is documented transitional coupling, not a pattern to extend. The generic
boundary is complete only when that parser/source projection is injected through
a nominal protocol.

Deployment maturity belongs to the packaged ``openhcs/omero`` bundle; web-client
application behaviour belongs to ``omero_openhcs``. Treat a web entry point as
compatible only when the installed OpenHCS distribution explicitly supports the
current ``PipelineConfig`` plus ``list[FunctionStep]`` declaration boundary.
OpenHCS does not infer that compatibility from package presence. Do not copy
credentials into pipeline source or assume that a remote OMERO plate is a local
directory.

Durable artifact materialization
--------------------------------

OpenHCS asks the selected PolyStore ``DataSink`` for contextual save arguments;
generic materialization code does not branch on OMERO or name its metadata
fields. ``OMEROLocalBackend`` resolves the base plate represented by the virtual
``images_dir`` and projects the parser and microscope declarations from its
cached ``PlateStructure``. Its own ``save_batch()`` then uses that context when
creating or updating a derived output plate.

The ``/omero/plate_<id>/...`` namespace is a virtual POSIX namespace, not a host
filesystem path. PolyStore normalizes it with ``PurePosixPath`` before parsing,
which preserves the same plate/output identity on Linux, macOS, and Windows.

Compiler contract
-----------------

An OMERO-backed source must provide the source workspace and metadata required
by the selected microscope/source declaration. Compilation then resolves normal
main-flow and named source bindings. Runtime workers access the configured
PolyStore backend; they do not open ad hoc OMERO connections based on path
strings.

Testing
-------

Keep unit tests at the owner boundary: use fake source references or backends
for compiler tests, PolyStore backend tests for generic I/O, and deployment
integration tests under ``tests/integration`` for the packaged
``openhcs/omero`` and ``omero_openhcs`` live-server behaviour. See
:doc:`../development/omero_testing`.

OMERO integration
=================

OMERO support crosses three ownership boundaries:

``omero_openhcs``
  Owns deployment, credentials, connection lifecycle, and the application-level
  OMERO integration package.

PolyStore
  Owns generic storage backends, virtual paths, source references, and ROI
  persistence primitives.

OpenHCS
  Owns microscope/source selection, source bindings, compilation, processing,
  and the desktop workflows that choose an OMERO source.

The current PolyStore ``OMEROLocalBackend`` still imports the OpenHCS
``FilenameParser`` registry while building its virtual source projection. This
is documented transitional coupling, not a pattern to extend. The generic
boundary is complete only when that parser/source projection is injected through
a nominal protocol.

The ``omero_openhcs`` web package is currently an alpha prototype. Its bundled
panel still emits an incomplete legacy execution request and its embedded
pipeline examples use removed imports, so it is not a supported current entry
point. Its development documentation lists the compatibility gate required
before deployment. Do not copy credentials into pipeline source or assume that
a remote OMERO plate is a local directory.

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
integration tests in ``omero_openhcs`` for live-server behavior. See
:doc:`../development/omero_testing`.

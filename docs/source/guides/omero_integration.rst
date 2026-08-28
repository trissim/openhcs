OMERO integration
=================

OMERO support crosses three ownership boundaries:

``omero_openhcs``
  Owns the OMERO.web application, templates, and application-level integration.
  Its source is bundled inside the OpenHCS distribution at
  ``openhcs/omero/plugin`` rather than installed from a repository-root project.

PolyStore
  Owns generic storage backends, virtual paths, source references, and ROI
  persistence primitives. Its OMERO declarations own text formats and MIME
  types, table parsing and service readiness, image-plane batching, and the
  canonical OMERO well and plane addresses.

OpenHCS
  Owns microscope/source selection, source bindings, compilation, processing,
  the desktop workflows that choose an OMERO source, and the packaged
  ``openhcs/omero`` deployment bundle and instance lifecycle. The lifecycle
  accepts a connection only after the PolyStore table-service declaration
  reports readiness; a responsive Blitz gateway alone is not the complete
  storage contract.

``OMEROLocalBackend`` generates and parses concrete virtual image identities
through its own ``OMEROPlaneAddress`` declaration. Pattern discovery projects
symbolic fields through PolyStore's matching ``OMEROPlaneFilenameTemplate``.
OpenHCS's ``OMEROFilenameParser`` maps both boundaries into
``FilenameParseResult``; neither package copies the filename grammar or imports
the other's registry.

Deployment maturity belongs to the packaged ``openhcs/omero`` bundle; web-client
application behaviour belongs to ``omero_openhcs``. Treat a web entry point as
compatible only when the installed OpenHCS distribution explicitly supports the
current ``PipelineConfig`` plus ``list[FunctionStep]`` declaration boundary.
OpenHCS does not infer that compatibility from package presence. Do not copy
credentials into pipeline source or assume that a remote OMERO plate is a local
directory.

The default Compose declaration starts the pinned upstream OMERO.web viewer. It
does not install or expose the alpha ``omero_openhcs`` panel. Connect through
``OMEROInstanceManager`` so the same packaged declaration and complete
gateway-plus-table-service readiness contract are used by desktop, test, and
packaged environments. The manager derives local connection defaults from that
declaration while allowing explicit host, port, web-port, user, and password
overrides for a remote instance. Start Docker before requesting the packaged
local stack. The manager waits briefly for an already-starting daemon to become
responsive, but host daemon lifecycle remains outside the OpenHCS integration
boundary. During a cold packaged start, repository creation and the OMERO table
component may finish independently. If the table component stops before the
repository exists, the manager waits for PolyStore's managed-repository
declaration, restarts that component within the packaged Compose stack, and
rechecks table readiness. Explicitly configured external stacks are never
restarted by this recovery path.

The current desktop alpha does not expose a supported OMERO management or
credential-entry window. The UI reference therefore records no OMERO desktop
surface. Configure and validate OMERO through the packaged instance manager and
the deployment and testing workflows documented here.

Durable artifact materialization
--------------------------------

OpenHCS asks the selected PolyStore ``DataSink`` for contextual save arguments;
generic materialization code does not branch on OMERO or name its metadata
fields. ``OMEROLocalBackend`` projects the virtual ``images_dir`` used to link
related artifacts. Its own ``save_batch()`` resolves image-plane coordinates
through ``OMEROPlaneAddress`` when creating or updating a derived output plate.

Analysis consolidation consumes CSV content from the execution ledger and asks
FileManager to write summaries through the compiled backend. It does not reopen
an OMERO virtual path as a local file. PolyStore's text-format members carry the
supported extension, MIME type, and table parser together, while its table
service checks OMERO's declared readiness and repository before creating a
table.

The ``/omero/plate_<id>/...`` namespace is a virtual POSIX namespace, not a host
filesystem path. PolyStore normalizes it with ``PurePosixPath`` before parsing,
which preserves the same plate/output identity on Linux, macOS, and Windows.
``OMEROWellAddress`` also supports multi-letter row labels used by plate formats
beyond 26 rows.

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
``openhcs/omero`` live-server behaviour. Test ``omero_openhcs`` separately
against its documented panel compatibility gate. See
:doc:`../development/omero_testing`.

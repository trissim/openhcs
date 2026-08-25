Extracted-package ownership
===========================

OpenHCS depends on eight first-party packages extracted from formerly generic
subsystems. Each package owns its reusable mechanism; OpenHCS owns the domain
declarations and adapters that compose those mechanisms into a microscopy
pipeline.

Ownership catalog
-----------------

ObjectState
  Owns lazy configuration types, context resolution, ObjectState editing,
  snapshots, provenance, undo, and generic projection/composition. OpenHCS owns
  its configuration hierarchy, compiler resolution boundary, and UI/code use.
  See :external+objectstate:doc:`ObjectState documentation <index>`.

ArrayBridge
  Owns memory-type taxonomy, callable memory metadata and decorators (including
  input, output, and execution memory), device-neutral array geometry, array
  conversion, stack utilities, device streams, and OOM behaviour. OpenHCS
  projects those declarations into callable contracts and owns semantic image
  shape roles, compiled conversion and framework-device plans, worker resource
  lifecycle, and resource policy. See
  :external+arraybridge:doc:`ArrayBridge documentation <index>`.

metaclass-registry
  Owns ``AutoRegisterMeta``, ``RegistryConfig``, ``RegistryFamily``, discovery,
  and generic caching. OpenHCS owns its nominal roots, declarations, and shared
  selection mixins. See
  :external+metaclass-registry:doc:`metaclass-registry documentation <index>`.

PolyStore
  Owns ``FileManager``, backend registration and lifecycle, generic storage
  formats, ROI, virtual-workspace references, storage/streaming primitives, and
  the generic ImageJ runtime declaration, Zarr compressor, chunk strategy, and
  backend configuration declarations. OpenHCS owns source-binding semantics,
  domain ``ImageFileFormat`` pixel/intensity metadata, artifact materialization
  plans, and a fieldless application registration subtype of PolyStore's
  ``ZarrConfig``. See
  :external+polystore:doc:`PolyStore documentation <index>`.

pyqt-reactive
  Owns generic reactive forms, widget protocols and strategies, managers,
  previews, reusable services, live Qt-window discovery and infrastructure, and
  Qt-rendered window snapshots. OpenHCS owns concrete workflows, Plate Manager,
  pipeline editors, application window identity and authorisation, and domain
  transport adapters. See
  :external+pyqt-reactive:doc:`pyqt-reactive documentation <index>`.

python-introspect
  Owns callable/signature analysis, wrapped-target resolution, parameter
  metadata, callable-declaration projection, and analyzer extensions. OpenHCS
  owns domain exclusions, callable contracts, and step-editor consumption. See
  the `python-introspect repository
  <https://github.com/OpenHCSDev/python-introspect>`_.

ZMQRuntime
  Owns generic process lifecycle, exact spawned-process ownership groups,
  automatic child reaping, request/status/progress protocols, typed connection
  and operation cancellation, and viewer-control transport.
  OpenHCS owns mapping compiled execution, source axes, runtime observations,
  and application services onto those protocols. See the `ZMQRuntime repository
  <https://github.com/OpenHCSDev/zmqruntime>`_.

pycodify
  Owns Python-source serialization, formatter registration, import construction,
  and collision handling. OpenHCS owns formatters and round-trip policies for its
  domain declarations. See the
  `pycodify repository <https://github.com/OpenHCSDev/pycodify>`_.

Integration-page standard
-------------------------

An OpenHCS integration page should contain:

1. the external owner and its responsibility;
2. the exact OpenHCS declaration or adapter surface;
3. the end-to-end integration sequence;
4. cross-package invariants and failure boundaries;
5. the compatibility constraint from ``pyproject.toml`` and canonical owner
   documentation.

It should not reproduce the external package's internal class architecture.

Migration policy
----------------

When a generic OpenHCS page moves to an owner package:

- publish and validate the owner page first;
- replace the old OpenHCS URL with a temporary transition page;
- link the owner narrative/API through first-party intersphinx inventories;
- link the separate OpenHCS integration page;
- remove the transition page from active navigation after one release.

Links into ``external/...`` are checkout-local and must not appear in published
documentation.

OMERO boundary
--------------

``omero_openhcs`` owns deployment and application integration. Generic OMERO
storage, ROI, address resolution, and backend behavior belong to PolyStore.

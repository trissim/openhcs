Microscope handler integration
==============================

``MicroscopeHandler`` is the nominal root for microscope/source-specific
behaviour. Concrete handlers self-register through ``AutoRegisterMeta`` and are
loaded by package discovery. ``create_microscope_handler`` selects an explicit
handler or performs ordered detection through the registered family.

Owned declarations
------------------

A handler owns:

- its microscope identity and detection behaviour;
- filename parsing and metadata-handler type;
- virtual-workspace root and source projection;
- compatible storage backends;
- viewer metadata required for component-aware display.

The handler registry is the source of truth. Generic discovery, compiler, and
UI code iterate ``MicroscopeHandler.__registry__.values()`` or the registry
service. They must not maintain a copied microscope list or import concrete
handler modules to inspect names.

The factory binds the root-declared ``plate_folder`` context uniformly after a
registered leaf is created. Leaf-specific initialisation belongs on that
leaf's ``create`` hook; the factory does not dispatch on concrete handler
names.

Source and compiler boundary
----------------------------

After initialisation, the handler supplies plate metadata and source files to
the source-workspace projection. The compiler uses that projection to construct
source universes, execution-axis values, main-flow dependencies, and named
source bindings. PolyStore owns the underlying references, virtual paths, and
backend I/O.

Extension boundary
------------------

A new microscope integration is a registered handler leaf with its parser,
metadata handler, identity, detection policy, and backend compatibility at the
owning module. Generic discovery and compilation consume that declaration.
The task-oriented procedure and required tests are in
:doc:`../development/source_binding_extension`; the ownership rule is expanded
in :doc:`nominal_ownership`.

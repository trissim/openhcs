Microscope handler integration
==============================

``MicroscopeHandler`` is the nominal root for microscope/source-specific
behavior. Concrete handlers self-register through ``AutoRegisterMeta`` and are
loaded by package discovery. ``create_microscope_handler`` selects an explicit
handler or performs ordered detection through the registered family.

Owned declarations
------------------

A handler owns:

- its microscope identity and detection behavior;
- filename parsing and metadata-handler type;
- virtual-workspace root and source projection;
- compatible storage backends;
- viewer metadata required for component-aware display.

The handler registry is the source of truth. Generic discovery, compiler, and
UI code iterate ``MicroscopeHandler.__registry__.values()`` or the registry
service. They must not maintain a copied microscope list or import concrete
handler modules to inspect names.

Source and compiler boundary
----------------------------

After initialization, the handler supplies plate metadata and source files to
the source-workspace projection. The compiler uses that projection to construct
source universes, execution-axis values, main-flow dependencies, and named
source bindings. PolyStore owns the underlying references, virtual paths, and
backend I/O.

Extension workflow
------------------

Add a concrete handler, parser, and metadata handler at the owning module. Set
the nominal registry attributes and declare backend compatibility. Test
explicit construction, auto-detection order, metadata projection, and source
binding compilation. See :doc:`../development/source_binding_extension` and
:doc:`nominal_ownership`.


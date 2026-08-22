Repository and module ownership
===============================

OpenHCS is an integration repository, not a monolith. Module boundaries follow
semantic ownership and eight generic foundations have been extracted into
first-party packages.

OpenHCS-owned areas
-------------------

``openhcs/core``
  Pipeline declarations, source/artifact semantics, compiler plans, runtime
  values, contexts, orchestrator, progress, and equivalence.

``openhcs/processing``
  Processing-library discovery, callable decorators/contracts, native and
  CellProfiler-compatible processing implementations.

``openhcs/interop``
  Translation boundaries such as CellProfiler parsing, module declarations,
  settings binding, and public lowering.

``openhcs/microscopes``
  Microscope-specific plate layout and metadata integration.

``openhcs/pyqt_gui``
  OpenHCS screens, workflow services, editors, Plate Manager, and adapters over
  pyqt-reactive.

``openhcs/runtime`` and ``openhcs/agent``
  Application-specific local/remote execution wiring and agent/UI services over
  the generic ZMQRuntime/MCP foundations.

Extracted foundations
---------------------

The submodules under ``external/`` own reusable behavior:

- ObjectState — lazy config and state/provenance
- ArrayBridge — array conversion, callable memory metadata, and framework-local
  device and execution mechanics
- PolyStore — storage, formats, ROI, and virtual workspace
- metaclass-registry — generic nominal registry mechanics
- pyqt-reactive — generic forms/widgets/services
- python-introspect — callable/signature analysis and declaration projection
- ZMQRuntime — generic execution and transport protocols
- pycodify — Python-source serialization

The root ``pyproject.toml`` declares their published distributions. Development
checkouts install the submodules explicitly; see ``docs/development_setup.md``.

Dependency direction
--------------------

Generic packages must not import OpenHCS domain code. OpenHCS adapters import
their generic interfaces and attach domain declarations. The current PolyStore
``OMEROLocalBackend`` parser lookup is a documented transitional violation of
this rule and must move to an injected nominal protocol. Generic core code must
not import a concrete processing backend merely to discover module names,
features, or behavior.

Registration
------------

Many nominal roots register subclasses through ``AutoRegisterMeta`` at import or
discovery time. Consumers query the root registry or a shared strategy mixin.
OpenHCS does not use a central ``initialize()`` call or a manually synchronized
``registries/`` directory.

Public declarations
-------------------

The primary pipeline boundary is ``PipelineConfig`` plus
``list[FunctionStep]``. Not every internal module is a stable public API. Prefer
the documented declaration and integration surfaces; owner-package internals
are documented by the owning package.

See :doc:`../architecture/nominal_ownership` and
:doc:`../architecture/external_foundations`.

External integrations
=====================

OpenHCS integrates external tools by lowering them into public declarations or
by adapting them at a typed runtime boundary. An integration must not create a
parallel pipeline, source, artifact, or execution model.

CellProfiler
------------

The ``.cppipe`` importer parses CellProfiler modules and lowers them to
``PipelineConfig`` plus ``FunctionStep`` declarations. Module declarations own
their callable and artifact semantics. See :doc:`cellprofiler_interop`.

Microscope and storage sources
------------------------------

Registered ``MicroscopeHandler`` implementations connect filename/metadata
semantics to PolyStore source references and virtual workspaces. OpenHCS owns
source selection and compiler bindings; PolyStore owns generic storage. See
:doc:`microscope_handler_integration` and :doc:`source_model`.

Viewers and remote execution
----------------------------

ZMQRuntime owns generic transport, execution lifecycle, progress, cancellation,
and viewer process primitives. PolyStore owns generic streaming backends.
OpenHCS owns execution payloads, progress projection, streaming declarations,
and Napari/Fiji adapters. See :doc:`streaming_boundary_and_wrappers`.

OMERO
-----

``omero_openhcs`` owns deployment and application integration. PolyStore owns
generic OMERO storage primitives. OpenHCS consumes the resulting source/backend
through normal source binding and compilation. See
:doc:`../guides/omero_integration`.

Boundary rule
-------------

Concrete integration details belong on their module, artifact, source,
measurement, or strategy declaration. Generic OpenHCS code queries those
authorities or their nominal registries and never imports every integration to
learn names or capabilities.

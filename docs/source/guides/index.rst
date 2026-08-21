Integration boundaries
======================

These explanations connect public pipeline declarations to the compiler,
storage, viewers, and optional deployment packages. Generic mechanics live
with the first-party package that owns them; these pages explain the OpenHCS
boundary rather than prescribing a task sequence.

.. toctree::
   :maxdepth: 2

   pipeline_compilation_workflow
   memory_type_integration
   viewer_management
   fiji_viewer_management
   omero_integration

Choose an explanation
---------------------

``pipeline_compilation_workflow``
  Why compilation is a declaration-to-runtime boundary and which authorities
  participate.

``memory_type_integration``
  How ArrayBridge input, output, and execution metadata enters callable
  contracts and framework-local worker planning.

``viewer_management`` and ``fiji_viewer_management``
  OpenHCS streaming configuration and the boundary with ZMQRuntime.

``omero_integration``
  Ownership split among OpenHCS, ``omero_openhcs``, and PolyStore.

For the underlying architecture, start with
:doc:`../architecture/system_overview`. For API imports, use
:doc:`../api/index`. For current examples and recipe lookup, use
:doc:`../reference/index`.

Integration guides
==================

These guides connect the public pipeline declarations to the compiler, storage,
viewers, and optional deployment packages. Generic mechanics live with the
first-party package that owns them; these pages document the OpenHCS boundary.

.. toctree::
   :maxdepth: 2

   complete_examples
   example_corpus_map
   pipeline_compilation_workflow
   memory_type_integration
   viewer_management
   fiji_viewer_management
   omero_integration
   testing_guide

Choose a guide
--------------

``complete_examples``
  Current declaration patterns and the source-of-truth example corpus.

``pipeline_compilation_workflow``
  What compilation accepts, produces, and validates before execution.

``memory_type_integration``
  How ArrayBridge metadata enters callable contracts and worker planning.

``viewer_management`` and ``fiji_viewer_management``
  OpenHCS streaming configuration and the boundary with ZMQRuntime.

``omero_integration``
  Ownership split among OpenHCS, ``omero_openhcs``, and PolyStore.

For the underlying architecture, start with
:doc:`../architecture/system_overview`. For API imports, use
:doc:`../api/index`.

Real-time visualization
=======================

OpenHCS streams compiled step outputs to Napari or Fiji. Streaming is an
independent step configuration that compiles into transport/viewer plans; it is
not artifact materialization, a special pipeline class, or a direct viewer call.

1. Install ``openhcs[gui,napari]``, ``openhcs[gui,fiji]``, or
   ``openhcs[gui,viz]``.
2. Open a step in the Pipeline Editor.
3. Enable its viewer configuration and choose persistence, batching, well
   filters, and display options.
4. Compile. OpenHCS validates source metadata, transport, and viewer
   requirements independently of artifact materialization.
5. Run the compiled plate and inspect the detached viewer.

To display only one step, leave viewer streaming disabled at global and
pipeline scope and enable ``napari_streaming_config`` or
``fiji_streaming_config`` only on that ``FunctionStep``. Use its ``well_filter``
to bound the diagnostic run and ``persistent=True`` to keep the detached viewer
available after completion. Recompile after any change.

This is distinct from ``step_materialization_config``, which saves the step's
ordinary main-flow result, and from typed artifact materialization, which owns
named images, labels, measurements, relationships, tables, grids, and external
resources. The compiled artifact plan is the authority for runtime-only versus
persistent outputs.

The compiler determines component order and attaches typed metadata to each
viewer batch. If a viewer does not start, check the execution log for dependency,
port/readiness, or Java/ImageJ errors before changing the pipeline.

See :doc:`../guides/viewer_management` and
:doc:`../guides/fiji_viewer_management` for lifecycle and ownership details.

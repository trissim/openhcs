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

Plan at least one final result layer around the scientific question the viewer
must answer. Intermediate source images, masks, object labels, and skeletons
are valuable for debugging, but they are not automatically a user-facing final
result. If the analysis assigns one object to another, the final image or label
artifact should encode that assignment directly--for example, by giving each
cell body and its assigned neurites the same stable object or label identity.
The viewer may derive matching display colors from that identity; separate body
and global-skeleton layers do not reveal the relationship.

The final interpretation should be a callable-owned typed artifact streamed or
materialized through its compiled plan, not a relationship invented only in the
viewer. Keep intermediate layers when they help explain the analysis, and add
the final relationship/result layer needed to judge the outcome.

Visibility, selection, zoom, and screenshots belong to the interactive viewer
and may change while another person or agent inspects the run. Use raw viewer
payloads, label identities, ROI summaries, and bounded image samples to verify
what the pipeline actually produced. Screenshots are evidence about rendering,
not the authority for whether a hidden artifact exists or which object identity
its pixels carry.

For a focused agent review, isolate the required image and ROI routes in one
viewer command, select the route that owns the active axes, and provide any
route-local axis indices in that same request. Napari validates the complete
layer set and navigation before changing visibility or selection, so a rejected
request does not leave a partial presentation. Use the ROI-summary command with
an explicit ``max_rois`` bound for metadata examples. That bound applies across
the complete response and the summary omits geometry; request raw viewer
payloads only when coordinates are needed.

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

When several steps stream into Napari, their layers share the component layout
declared by the display configuration. Slider sizes come from the source
component domains. A step that has reduced channel, Z, time, or another display
component occupies a singleton slot for that component while full-resolution
layers retain the complete domain. If one axis unexpectedly takes another
axis's size, record the layer shapes and axis labels; that indicates a semantic
axis-projection failure rather than a property of the input plate.

The dimension-label overlay follows the selected OpenHCS layer. Hiding another
layer should not change its labels, and selecting a non-OpenHCS layer should
clear them. A mismatch between the selected layer and the overlay is a viewer
route-state defect, not evidence that the source axes changed.

After the last batch is accepted, OpenHCS waits for the viewer to settle before
capturing state or closing a non-persistent viewer. Napari reports incremental
completed/total layer-update progress while yielding between routes on the Qt
event loop. The settlement deadline measures time without forward progress, not
the total time required for a large transfer. A route exception or a genuinely
stalled update fails execution instead of being hidden behind a successful ZMQ
acknowledgment.

A layer count or nonzero-pixel check verifies delivery, not scientific
adequacy. Viewer validation should also confirm that the final layer visibly
answers the user's question.

See :doc:`../guides/viewer_management` and
:doc:`../guides/fiji_viewer_management` for lifecycle and ownership details.

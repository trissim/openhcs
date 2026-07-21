Viewer streaming
================

OpenHCS declares viewer streaming on a step independently of artifact
materialization. Compilation lowers enabled streaming configuration into typed
transport and viewer plans. PolyStore owns the
streaming backend primitives, while ZMQRuntime owns generic process, socket,
acknowledgment, and viewer-lifecycle machinery.

OpenHCS keeps only application semantics: supported viewer identities, step
configuration, metadata projection, display policy, and Napari/Fiji adapters.
Those declarations are rooted in ``StreamingConfig`` and the registered
streaming configuration types; generic code should iterate that registry rather
than maintain a viewer-name table.

Desktop workflow
----------------

1. Install ``openhcs[gui,napari]`` or ``openhcs[gui,viz]``.
2. Open a step in the Pipeline Editor.
3. Enable the Napari streaming configuration and choose its display options.
4. Compile the plate. Port, backend, and source-metadata requirements are
   validated before execution.
5. Run the compiled selection. The viewer process is managed outside the worker
   that emits image batches.

Persistence and reuse are configuration policies, not guarantees that an
arbitrary process on the same port is compatible. Readiness uses the typed
control protocol before image data is sent.

Execution completion also has a typed viewer boundary. Napari drains queued
layer routes incrementally on the Qt thread and reports completed/total update
counts plus the active route. The caller renews its deadline whenever that
count advances, so the configured interval means "no progress for this long"
rather than "the whole viewer must finish this quickly." A route failure or a
stalled count is an execution failure; a successful transport acknowledgment
alone is not evidence that the corresponding layer was rendered.

Choosing one step to inspect
----------------------------

Viewer settings are inherited, but efficient inspection starts by constraining
the execution domain rather than only the viewer:

1. set ``pipeline_config.well_filter_config.well_filter`` to the diagnostic
   wells (for example, ``"B03"``), so other wells are not loaded or processed;
2. keep the global and pipeline viewer ``enabled`` defaults false;
3. set ``napari_streaming_config.enabled=True`` or
   ``fiji_streaming_config.enabled=True`` only on the target ``FunctionStep``;
4. leave the viewer ``well_filter`` inherited unless it must narrow the pipeline
   domain further;
5. use ``persistent=True`` when the viewer must remain available for structured
   inspection after the pipeline finishes; and
6. compile again, because viewer intent is lowered into the immutable step plan.

If the viewer is the only desired image destination, set
``pipeline_config.path_planning_config.well_filter=0``. This suppresses the
automatic final main-flow output plate while leaving the in-memory value
available for streaming. Explicit step checkpoints and typed named-artifact
materializations remain independent and must be disabled separately when they
are not wanted.

Designing the final view for the user
-------------------------------------

Choose streamed layers from the question the user must answer, not only from
the steps that are convenient to debug. Source images, threshold masks,
segmentations, and skeletons are useful intermediate evidence. They do not
replace a final result layer that makes the analysis conclusion visible.

When the result is a relationship, the final artifact must encode that
relationship directly. For example, a neurite-assignment result should give a
cell body and its assigned neurites the same stable object or label identity;
the viewer may derive matching display colors from that identity. A cell-body
layer beside a global skeleton layer proves that both stages ran, but it does
not show which neurites belong to which cell. Similar requirements apply to
parent/child objects, tracks, neighborhoods, and class assignments.

Produce the interpretation as a callable-owned typed image or label artifact,
then stream or materialize it through the compiled artifact plan. Viewer-only
annotations must not become the semantic authority for a scientific result.
Keep useful intermediate layers for diagnosis, but consider viewer review
incomplete until the final layer lets the user verify the requested conclusion.

An interactive viewer has two evidence boundaries. Layer visibility, selection,
navigation, zoom, and screenshots are presentation state that the user may
change while an agent is working. Use raw route payloads, object-label IDs, ROI
summaries, and bounded image samples to establish what the pipeline produced.
Use screenshots to assess rendering and ergonomics only; a hidden layer is not
an absent artifact, and screenshot colors must not override typed object
identity in the payload.

For example, a reviewed pipeline document can place the override directly on
the step being inspected:

.. code-block:: python

   FunctionStep(
       name="Inspect segmented cells",
       func=segment_cells,
       napari_streaming_config=LazyNapariStreamingConfig(
           enabled=True,
           persistent=True,
       ),
   )

Use the current reflected step schema rather than treating that example as a
field inventory. Through MCP, request ``openhcs_describe_config_schema`` with
``config_type="step"`` and one returned nested path:
``napari_streaming_config``, ``fiji_streaming_config``, or
``step_materialization_config``.

Streaming, checkpointing, and named artifacts
----------------------------------------------

These mechanisms answer different questions:

``NapariStreamingConfig`` / ``FijiStreamingConfig``
  Show eligible outputs while the selected step executes. Display axes,
  batching, transport, viewer persistence, and well selection belong here.

``StepMaterializationConfig``
  Save the step's ordinary main-flow result as a persistent checkpoint. Set it
  on the exact step whose main-flow output is needed. It does not persist every
  named artifact produced by the callable.

Typed artifact materialization
  Persists named image, label, measurement, relationship, table, grid, or
  external-resource outputs according to the callable-owned artifact contract
  and compiled runtime-artifact materialization plan.

Paused runtime inspection
  Shows invocation parameters, runtime-value records, and artifact references
  from an active debug worker. It is runtime evidence, not a persistence policy
  and not a replacement for visual validation.

Inspect the compiled artifact plan before execution to see which outputs are
runtime-only and which have persistent targets. After execution, use viewer
state, payload, image-sample, and ROI-summary tools for concrete visual
evidence; a successfully launched viewer alone is not result validation.
Likewise, layer existence and nonzero pixels prove transport and content, not
that the chosen layers communicate the requested scientific result.

See :doc:`../architecture/streaming_boundary_and_wrappers` for ownership and
:doc:`fiji_viewer_management` for Fiji-specific requirements.

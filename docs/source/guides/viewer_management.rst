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

ROI inspection and cropping
---------------------------

The ``openhcs[napari]`` installation includes both ``napari-roi-manager`` and
``napari-crop``. The ``viz`` and ``all`` extras include the same Napari
surface, so ROI inspection and cropping do not require a second plugin install.

OpenHCS streams ROI artifacts through the registered ``SHAPES`` display
boundary as native N-dimensional Napari Shapes layers. Each member retains its
stable ``label`` feature and scalarized source metadata. Consequently the same
layer can be selected, edited with Napari's Shapes controls, inspected as ROI
geometry, or supplied to the crop plugin without a callable-specific viewer
adapter. The viewer's built-in feature table reads those Shapes features
directly and is opened by default with every OpenHCS Napari viewer. Selecting a
row selects that member directly on the same Shapes layer; there is no
projected or synchronized copy. The table follows whichever feature-bearing
layer is selected in the layer list.

Fresh OpenHCS Napari windows use the available desktop geometry and place the
feature table in a full-width lower dock. This leaves a useful image canvas and
table visible together without depending on fixed screen coordinates. The ROI
Manager remains an optional independent workspace; opening or closing it does
not replace the streamed Shapes layer or the table that owns its features.

The ``napari-roi-manager`` plugin currently owns a separate fixed-2D Shapes
layer and does not attach its table to an existing N-dimensional Shapes layer.
OpenHCS therefore does not mirror streamed geometry into that private layer.
Use ``Plugins > napari-roi-manager > ROI Manager`` when an independent ImageJ
ROI import/export workspace is useful, and use
``Plugins > napari-crop > Crop Region(s)`` to crop an image directly from the
authoritative streamed Shapes geometry.

Dense segmentation masks remain Napari Labels layers. If a downstream workflow
needs editable contours or paths, stream or materialize the callable-owned ROI
artifact rather than inferring object identity from a screenshot.

Spatial graphs and neuronal morphology
--------------------------------------

A skeleton mask records occupied pixels; it does not preserve nodes, directed
edges, parentage, or branch measurements. Callables whose scientific result is
path topology should therefore declare a ``SpatialGraphArtifactType`` and return
one ``SpatialGraph`` containing the authoritative nodes, paths, and scalar edge
features.

The same graph can have multiple format projections without duplicating the
analysis. ``SWCOptions`` writes a directed acyclic morphology forest as standard
SWC. ``SpatialGraphROIOptions`` writes a 2-D ``.graph.roi.zip`` projection whose
polyline members retain graph/node identities and branch features. Viewer
capability routing selects that ROI projection for Napari automatically, where
it appears as a native path Shapes layer. Select that layer to see branch
distance, Euclidean distance, tortuosity, distance from the soma, branch type,
and neuron identity in the feature table. Selecting a row selects the exact
rendered branch.

Saved ``.swc`` files are viewer-readable too. The OpenHCS Napari plugin
registers a standard SWC reader and opens the physical morphology as 3-D sample
Points plus parent-child Shapes. Both layers retain the standard sample ID,
structure type, radius, and parent ID columns. Fiji users can open the same SWC
through Fiji's SNT morphology support. Standard SWC has no field for arbitrary
OpenHCS edge measurements, so use the ``.graph.roi.zip`` projection when the
full branch-feature table is the important review surface. Live pipeline
viewing projects the in-memory graph directly; it does not serialize and parse
SWC first.

SWC materialization rejects cyclic or multiple-parent graphs. A generic spatial
graph may still represent a cyclic assay, but it must use a format that can
preserve that topology rather than silently losing edges through SWC. The ROI
projection is a visualization/interchange view; the ``SpatialGraph`` remains
the semantic owner.

Execution completion also has a typed viewer boundary. Napari drains queued
layer routes incrementally on the Qt thread and reports completed/total update
counts, the active route, completed bounded work units within that route, and
whether one native work unit is currently executing. Control transport owns its
socket independently of Qt, so settlement remains observable while Napari is
triangulating a complex Shapes member. The caller renews its no-progress
deadline when a route or work-unit count advances and does not misclassify a
declared active native mutation as an idle viewer. A route failure or a route
that neither advances nor executes declared work is an execution failure; a
successful transport acknowledgment alone is not evidence that the
corresponding layer was rendered.

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

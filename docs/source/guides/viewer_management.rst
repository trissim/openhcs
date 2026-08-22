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

For task instructions, including how to restrict a diagnostic run to one step
and a bounded set of wells, use :doc:`../user_guide/real_time_visualization`.
This page explains the ownership, lifecycle, evidence, and artifact boundaries
behind that workflow.

Persistence and reuse are configuration policies, not guarantees that an
arbitrary process on the same port is compatible. Readiness uses the typed
control protocol before image data is sent.

Display-axis ownership
----------------------

Napari uses the display configuration's component layout as the shared axis
contract for every layer in one viewer. The source component domains determine
the slider sizes and labels. A step that reduces a component, such as channel or
Z, contributes a singleton slot at that component's declared position; it does
not remove the slot or replace another layer's domain size. Images, points, ROI
shapes, translations, and exact-slice navigation all use those same semantic
slots. Route-local batches therefore cannot make unrelated axes inherit their
cardinality.

The selected OpenHCS layer route owns the dimension-label overlay. Changing the
visibility of another route does not transfer that authority. Selecting a
different OpenHCS route refreshes the overlay from that route's semantic axes;
selecting a non-OpenHCS layer clears the OpenHCS route and overlay instead of
leaving stale labels visible.

ROI inspection and cropping
---------------------------

The ``openhcs[napari]`` installation includes OpenHCS's first-party ROI Manager
and ``napari-crop``. The ``viz`` and ``all`` extras include the same Napari
surface, so ROI inspection and cropping do not require a second plugin install
or a separately published ROI-manager distribution.

OpenHCS streams ROI artifacts through the registered ``SHAPES`` display
boundary as native N-dimensional Napari Shapes layers. Each member retains its
stable ``label`` feature and scalarized source metadata. Consequently the same
layer can be selected, edited with Napari's Shapes controls, inspected as ROI
geometry, or supplied to the crop plugin without a callable-specific viewer
adapter. OpenHCS's first-party ROI Manager reads those Shapes features directly
and is opened by default with every OpenHCS Napari viewer. Selecting a row
selects that member directly on the same Shapes layer; there is no projected or
synchronized copy. The manager follows whichever Shapes layer is selected in
the layer list.

For streamed result layers, a table-row selection also reveals and activates
the row's owning layer and moves every non-displayed viewer axis to that
member's native N-dimensional coordinates. This remains unambiguous when
several ROI layers were selected: the row's authoritative Shapes layer becomes
the active layer without changing the user-defined layer-stack order, and
restores the native selection after the slice change so its outline stays
visible. An ``OpenHCS ROI selection`` toolbar beside this workflow exposes
``Selected ROI outline`` from 1 to 10 pixels plus synchronized ``Selection
color``, ``ROI group color``, and ``ROI layer color`` buttons. Selection color
controls Napari's native, Preferences-backed global highlight. ROI group color
changes every native ROI member linked to the currently selected object while
preserving other groups; ROI layer color assigns the active result layer's
native edge-color property uniformly across every ROI. OpenHCS
replaces only Napari's untouched stock cyan highlight with a high-contrast
yellow default; any user-selected color remains authoritative.

Fresh OpenHCS Napari windows use the available desktop geometry and place the
ROI Manager table in a full-width lower dock. This leaves a useful image canvas
and table visible together without depending on fixed screen coordinates.

The OpenHCS ROI Manager is mounted when the viewer starts and binds to the first
streamed ROI result when it arrives. It can also be opened through
``Plugins > OpenHCS > OpenHCS ROI Manager``. Opening the
manager does not create a layer: its rows, shape types, feature columns, and
selection are live projections of the active native Shapes layer. Select a
different Shapes layer in Napari's layer list and the manager reconnects to
that owner. Its Fiji-style Add/Register, Remove, Rename, Specify, Load, Save,
and Show All actions therefore edit that same layer instead of a private ROI
copy. With Show All disabled, the selected ROI keeps Napari's visible native
highlight while the unselected base outlines are hidden; layer opacity and
user-assigned colors are restored unchanged. Table selection writes the
layer's native ``selected_data``, so the same exact-slice navigation and
visible highlight used by OpenHCS viewer navigation apply. Creating an empty ROI
set is an explicit manager action; its dimensionality follows the current viewer.
Use
``Plugins > napari-crop > Crop Region(s)`` to crop an image directly from the
same authoritative Shapes geometry.

The first-party widget incorporates the native-Shapes implementation reviewed
at ``OpenHCSDev/openhcs-napari-roi-manager`` tag ``v0.0.7``. OpenHCS preserves
the upstream authorship and BSD-3-Clause license in the wheel's
``THIRD_PARTY_LICENSES/napari-roi-manager-LICENSE`` artifact.

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
and neuron identity in the ROI table. Selecting a row selects the exact
rendered branch. When the graph output declares an object-member subject,
selection expands to every branch owned by that object and to the one linked
aggregate-measurement row. The branch rows remain separate and retain their
edge metrics; OpenHCS does not fabricate one disconnected polygon to represent
the neuron. Framework linkage keys live in native layer metadata rather than
cluttering the biological result table.

Saved ``.swc`` files are viewer-readable too. The OpenHCS Napari plugin
registers a standard SWC reader and opens the physical morphology as 3-D sample
Points plus parent-child Shapes. Both layers retain the standard sample ID,
structure type, radius, and parent ID columns. Fiji users can open the same SWC
through Fiji's SNT morphology support. Standard SWC has no field for arbitrary
OpenHCS edge measurements, so use the ``.graph.roi.zip`` projection when the
full branch-feature table in the ROI Manager is the important review surface. Live pipeline
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

Standalone review of already persisted plate files launches through the
validated OpenHCS UI process identity when a UI bridge is available. The
platform authority projects only its declared graphical-session and child
process variables, preserving display credentials without exposing unrelated
GUI environment values to Napari, Fiji, or their plugins. Explicit file paths
are resolved from the unified plate inventory independently of the default
image filter; filtered queries report that an existing record was excluded and
name its actual kind. If a detached viewer fails during startup, the structured
error contains the durable launch-log path and a bounded tail of that log so
the Qt, Fiji, or transport cause is visible without source inspection.

See :doc:`../architecture/streaming_boundary_and_wrappers` for ownership and
:doc:`fiji_viewer_management` for Fiji-specific requirements.

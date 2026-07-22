ROI and storage integration
===========================

ROI handling crosses semantic, persistence, and presentation boundaries. No
single viewer or file format owns all three.

Ownership
---------

OpenHCS
  Owns object-label and spatial-graph artifact declarations, compiled
  materialization plans, source-plane and object-subject identity, automatic
  semantic-to-ROI projection, and the Napari/Fiji adapters.

`PolyStore <https://github.com/OpenHCSDev/PolyStore>`_
  Owns reusable ROI value types, ImageJ ROI archive parsing/writing, feature
  sidecars, backend storage, virtual paths, and transport payload mechanics.

Napari
  Owns mounted ``Shapes`` geometry, feature rows, native selection, visibility,
  edge color, layer order, and N-dimensional viewer coordinates.

``openhcs.napari_roi_manager``
  Owns the Fiji-style manager interaction over the active native Napari Shapes
  layer as part of the existing ``openhcs`` Napari plugin. Its widget is a
  projection of that layer, not a second geometry or selection store. The
  retained upstream BSD-3-Clause notice is packaged with OpenHCS.

See :doc:`external_foundations` for the extracted-package policy and
:doc:`streaming_boundary_and_wrappers` for the transport/viewer split.

Compiled materialization
------------------------

An ROI archive is a materialization of a typed artifact; it is not the
artifact's semantic identity. ``ArtifactOutputPlan`` retains the exact artifact
type, producer, execution scope, source relations, materialization options, and
object-subject relations. Runtime materialization gives the registered writer
that complete plan.

Object-label artifacts use their nominal materialization strategy to project
label members into ROI shapes. Spatial graphs have two independent registered
projections:

``SpatialGraphROIOptions``
  Writes one feature-bearing polyline ROI per graph edge. It retains edge
  identity and scalar branch features. ImageJ ROI geometry is two-dimensional,
  so a genuinely three-dimensional graph must use SWC instead of silently
  dropping an axis.

``SWCOptions``
  Writes a standard morphology forest with deterministic sample/parent IDs,
  physical coordinates, radii, and biological structure types. SWC exchange
  does not replace the feature-bearing ROI projection because standard SWC
  cannot carry the same branch and object-linkage table.

Both projections consume the nominal ``SpatialGraph`` owner. A writer or viewer
must not rebuild topology by matching filenames, row order, or displayed layer
titles.

Plane and feature identity
--------------------------

ROI geometry may be multipart and may represent object identifiers outside a
``uint16`` label range. The declared label/object identity therefore travels as
metadata rather than being inferred from pixel dtype or one geometry member.
ImageJ Z/T positions and OpenHCS source-plane provenance remain distinct from
the two coordinate columns stored by the ROI shape itself.

Feature sidecars preserve arbitrary JSON-compatible ROI feature values through
PolyStore. Framework-owned cross-artifact linkage uses the
``ObjectArtifactSubjectBinding`` metadata projected from exact
``ArtifactSpecRelation`` declarations. Those keys remain hidden from the
biological feature table. See :doc:`artifact_contract_system` for the relation
and subject-identity path.

Native Napari projection
------------------------

The Napari adapter mounts streamed ROI values as native N-dimensional Shapes
layers. Geometry, shape type, features, selection, colors, visibility, and
layer order remain on those exact layer objects. Selecting a feature row can
activate the owning layer and navigate non-displayed axes to the member's
declared coordinates; it does not reorder the layer stack.

When sibling outputs declare the same object subject, selection can expand from
one native path to all members of that object and its aggregate measurement row.
The join uses the producer-scoped subject token and declared local identifiers,
not a hardcoded assay column such as ``cell`` or ``label``. Disconnected paths
remain separate geometry members even when they represent one biological
object.

The OpenHCS ROI Manager binds to the active Shapes layer through its public
``connect_layer()`` seam. Opening the manager does not create a private layer.
Renaming, removal, Show All, loading, saving, and row selection mutate or
project the native owner. Creating an empty ROI set is a separate explicit
action whose dimensionality follows the current viewer.

Extension rule
--------------

Add new semantic ROI behavior to the owning artifact type, materialization
option/writer, or ``ArtifactSpecRelation`` leaf. Add generic file-format and
backend mechanics to PolyStore. Add presentation behavior to the viewer/plugin
that owns the native UI state. Do not synchronize a private ROI mirror, infer
object identity from feature names, or put viewer-specific dispatch in the
compiler.

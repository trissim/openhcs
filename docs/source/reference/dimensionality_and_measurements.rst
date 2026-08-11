Dimensionality and measurement capabilities
============================================

Use this reference to distinguish plane-local processing from true volumetric
processing and to check the current measurement families. OpenHCS has no global
2D/3D switch. Dimensional behaviour belongs to each processing callable and its
typed inputs.

How dimensionality is declared
------------------------------

The effective execution domain is determined by these declarations:

``variable_components``
  Declares which microscopy components are assembled along the outer stack
  axis. Selecting Z describes the component identity; it does not by itself
  select volumetric processing.

``ProcessingContract``
  Declares whether the assembled runtime stack is processed plane by plane,
  as a whole, through an explicit flexible control, or as a volume collapsed
  to a plane.

Image and object-label input modes
  A callable can require the complete image payload or full-stack labels. These
  declarations preserve a nested volume when an outer runtime plane axis also
  exists.

Object-label domain
  ``PAYLOAD`` means one object-ID domain across the complete label payload.
  ``PLANE`` means independent object-ID domains for the outer planes.

Array dimensionality alone is not an execution contract.

Current segmentation boundary
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - Route
     - Current domain
     - Consequence
   * - CellProfiler-compatible ``Watershed``
     - Complete image payload
     - Accepts a 3D image domain and emits payload-scoped labels, so an object
       can span Z planes.
   * - ``IdentifyPrimaryObjects``
     - Plane-local
     - Each outer plane is segmented independently.
   * - ``IdentifySecondaryObjects``
     - Plane-local
     - Secondary labels remain independent between outer planes.
   * - ``IdentifyTertiaryObjects``
     - Plane-local
     - Tertiary labels remain independent between outer planes.
   * - Image and object morphology
     - Function and footprint dependent
     - Volumetric variants, including ball-footprint and 3D object operations,
       coexist with plane-local variants. Inspect the selected callable rather
       than assuming that the whole family shares one domain.

Plane-local labels are not stitched into volumetric objects. Use a route that
produces a payload-scoped label domain when biological identity must persist
through Z.

Current volumetric measurements
-------------------------------

``MeasureObjectIntensity`` accepts a payload-scoped 3D label domain and emits
one row per volumetric object. Its 3D fields include Z positions such as centre
of mass and maximum-intensity Z.

``MeasureObjectSizeShape`` accepts full-stack labels. For 3D objects it emits
the volumetric schema, including ``Volume``, ``SurfaceArea``, ``Center_Z``,
``BoundingBoxVolume``, Z bounds, extent, Euler number, axis lengths, and
equivalent diameter. It does not relabel those values as 2D area features.

Occupied-volume routes report occupied volume, surface area, and total volume;
physical surface-area scaling can use declared Z, Y, and X voxel spacing.

Measurement families
--------------------

The current CellProfiler-compatible catalogue includes these measurement
families:

- image and object intensity;
- 2D and 3D object size and shape;
- texture and granularity;
- radial intensity distribution and Zernike features;
- colocalisation;
- image and object overlap;
- neighbours, skeletons, counts, and object relationships;
- image quality;
- area and volume occupied; and
- calculated and classified outputs.

Availability of a family does not imply that every function or feature in that
family is volumetric. The selected callable's declaration and output schema are
the exact authority. Use the desktop function search or the MCP capability
search to inspect the callable available from the connected execution server.

Validation boundary
-------------------

The source-backed Official30 corpus includes a 3D monolayer pipeline with
volumetric Watershed segmentation followed by ``MeasureObjectIntensity`` and
``MeasureObjectSizeShape``. The acceptance suite compares OpenHCS execution
with native CellProfiler reference outputs under the policy documented in
:doc:`../architecture/measurement_equivalence_system`.

The corpus is compatibility evidence for the included pipelines, not a claim
that every processing function has a 3D implementation. See
:doc:`../concepts/data_dimensions` for the dimensionality mental model.

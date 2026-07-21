Domain fundamentals
===================

High-content microscopy produces images indexed by experimental dimensions such
as plate, well, site, channel, Z plane, and time point. The analysis problem is
not only applying an image function: it is preserving the meaning of those
dimensions while selecting sources, constructing batches, routing outputs, and
recording measurements.

Why the data model matters
--------------------------

Microscope vendors differ in filename layout, metadata, and storage. OpenHCS
uses registered microscope handlers and PolyStore source references to project
those formats into a common source model. The original files remain the source;
the projection supplies typed component identities and provenance to the
compiler.

Why compilation matters
-----------------------

A pipeline declaration does not contain enough information to run safely. The
compiler combines it with a plate's metadata to resolve:

- which images satisfy each input;
- which component varies along an array axis;
- which work can run independently;
- callable locality and memory conversion;
- artifact dependencies and output materialization.

Failures at these boundaries are reported before runtime whenever possible.

Why typed results matter
------------------------

Image arrays are only one result kind. Segmentation labels, measurements,
relationships, tables, positions, and other artifacts carry semantics that
cannot be recovered reliably from a filename or array shape. OpenHCS keeps
those semantics in callable/module contracts, typed plans, and runtime values.

Start with :doc:`core_model`, :doc:`data_dimensions`, and
:doc:`pipelines_and_steps`. Performance depends on the dataset, hardware,
backend, and callable mix; use repository benchmarks rather than undocumented
speed or scale claims.

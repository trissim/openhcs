Loading and naming image sources
================================

An OpenHCS plate can contain one image store or several stores together. Two
separate decisions are involved:

1. a microscope or store handler owns **ingestion** and publishes addressable
   planes;
2. source bindings optionally **select and name** those planes for pipeline
   inputs.

For a recognized HCS layout, OpenHCS keeps the native microscope handler. CZI,
OME-TIFF, and other supported rich containers are decoded by the Bio-Formats
store handler using their embedded metadata. Add the containing directory to
the Plate Manager and initialize with auto-detection; a source binding is not
required merely to open or browse the store.

``SourceBindingsHandler`` has a different role: it is the ingestion fallback for
an arbitrary image folder whose layout is not recognized. A non-empty
``PipelineConfig.source_bindings_config`` tells that handler which ordinary
TIFF, PNG, JPEG, or other registered files form the source universe and how
filename metadata should be interpreted.

OpenHCS asks each image store to publish addressable planes before applying the
bindings. A binding selects those planes by declared component metadata or by
their original file provenance and gives them a stable name such as ``DNA`` or
``Mask``. Pipeline declarations do not traverse CZI, OME-TIFF, or OME-Zarr
internals and do not select a storage backend.

This means ``SourceBindingsConfig`` can still name channels or samples from a
CZI after Bio-Formats has decoded it. In that case the Bio-Formats handler remains
the ingestion owner; the binding is semantic selection, not a replacement CZI
loader.

Native layout versus complete metadata
----------------------------------------

Filename/layout recognition and automatic detection are related but distinct.
Run ``openhcs_inspect_plate_path`` with ``microscope_type="auto"`` first. Its
``format_specific_handler_candidates`` are derived live from the registered
``MicroscopeHandler`` parsers and detection contracts. If a broad store handler
was selected but a format-specific candidate recognizes every tested physical
filename under its declared ``root_dir``, do not assume the broad handler is the
better semantic owner merely because native metadata is missing.

Opera Phenix is a representative case. A complete plate export contains the
``Images/`` layout plus ``Index.xml``; that metadata enables native automatic
detection, grid geometry, pixel size, and other plate facts. A deliberately
small subset may contain correctly named files such as
``r04c09f11p01-ch1sk1fk1fl1.tiff`` under ``Images/`` without ``Index.xml``. For
native Opera Phenix semantics, obtain the complete export. The Opera Phenix
owner requires its ``Index.xml`` detection contract; matching filenames alone do
not make the subset a valid native plate. If those TIFFs are intentionally being
treated as loose ordinary images, select them with ``SourceBindingsConfig`` and
declare their Well/Site/Channel/Z/Time identities instead of explicitly selecting
``microscope_type="opera_phenix"`` or accepting a broad decoder's inferred sample
layout.

The folder-onboarding authoring context lists every currently registered handler
and its handler-owned selection role. Use the format-specific owner for recognized
vendor layouts, the broad structured-store owner for supported rich containers
without a stronger native match, and the declared-file fallback for arbitrary
ordinary files whose schema you provide.

``source_stack_components`` describes axes physically present inside one selected
file or store payload. Do not declare ``SITE`` or another source-stack axis when
each site is a separate ordinary 2-D file. Give each file its component identity,
then use ``processing_config.variable_components`` to assemble those selected files
along the callable's stack axis.

Coordinates and source identity
-------------------------------

Every resolved plane has one sample identity represented by ``WELL``. This is the
embedded plate well when the image metadata declares one; otherwise it is the
exact source-container identity. ``SITE``, ``CHANNEL``, ``Z_INDEX``, and
``TIMEPOINT`` identify the plane within that sample. If a store declares an axis
absent or singleton, its coordinate is ``"1"``. Embedded coordinates are never
replaced by filename guesses, and conflicting plane identities stop
initialization with an explicit error.

Each plane also retains a typed reference to its owning store. The image browser,
metadata browser, compiler, and runtime all consume the same resolved plate
projection. Saving a changed source-binding config invalidates that projection;
the next normal initialization rebuilds it and updates the available aliases and
coordinates without a separate UI metadata copy.

Executable code-mode declarations
---------------------------------

The following block constructs a complete Pipeline Editor code document for a
directory that mixes stores. The intermediate config values deliberately show
both source-binding roles:

- ``tiff_png_config`` can drive ``SourceBindingsHandler`` when the directory is
  an otherwise unrecognized collection of ordinary files;
- the CZI, OME-TIFF, and OME-Zarr configs name/select planes after their store
  handlers decode them.

Choose one as ``pipeline_config`` and keep the ``pipeline_steps`` assignment in
the same document. Filenames are exact examples; replace them with names present
directly under the selected plate directory.

.. code-block:: python

   from openhcs.constants.input_source import InputSource
   from openhcs.core.config import (
       LazyProcessingConfig,
       LazyStepSourceBindingsConfig,
       PipelineConfig,
   )
   from openhcs.core.source_bindings import (
       NamedSourceBinding,
       SourceBindingsConfig,
       SourceFilterClause,
       SourceFilterMatchType,
       SourceFilterSubject,
       SourceSelector,
   )
   from openhcs.core.steps.function_step import FunctionStep
   from openhcs.processing.backends.processors.numpy_processor import (
       stack_percentile_normalize,
   )


   def bind_file(alias: str, filename: str) -> NamedSourceBinding:
       return NamedSourceBinding(
           alias=alias,
           selector=SourceSelector(
               filters=(
                   SourceFilterClause(
                       subject=SourceFilterSubject.FILE,
                       match_type=SourceFilterMatchType.EQUALS,
                       value=filename,
                   ),
               ),
           ),
       )


   tiff_png_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(
               bind_file("DNA", "nuclei.tif"),
               bind_file("Mask", "segmentation.png"),
           ),
       ),
   )

   czi_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(bind_file("DNA", "experiment.czi"),),
       ),
   )

   ome_tiff_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(bind_file("DNA", "plate.ome.tif"),),
       ),
   )

   ome_zarr_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(bind_file("DNA", "plate.zarr"),),
       ),
   )

   mixed_store_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(
               bind_file("DNA", "plate.zarr"),
               bind_file("Brightfield", "brightfield.tif"),
               bind_file("Mask", "labels.png"),
           ),
       ),
   )

   pipeline_config = mixed_store_config

   pipeline_steps = [
       FunctionStep(
           name="Normalize DNA",
           func=stack_percentile_normalize,
           processing_config=LazyProcessingConfig(
               input_source=InputSource.PIPELINE_START,
           ),
           source_bindings=LazyStepSourceBindingsConfig(
               enabled=True,
               bindings=(NamedSourceBinding(alias="DNA"),),
           ),
       ),
   ]

After applying the code, initialize the selected plate normally with auto-
detection. OpenHCS first chooses the native, Bio-Formats/store, or arbitrary-
folder ingestion owner, then resolves every binding over the planes that owner
published. CZI and embedded OME metadata require the Bio-Formats runtime included
by an OpenHCS installation that enables those formats. Do not force a CZI or OME
container through ``SourceBindingsHandler`` when its structured decoder is
missing or unhealthy; repair that decoder instead.

Selecting channels and samples
------------------------------

Use component selectors when the store already declares the coordinate you need.
This config fragment names channel ``2`` from well ``A01`` without encoding
either value in a format-specific path rule. Use it as ``pipeline_config`` in
the complete document above; source-backed MCP execution and Pipeline Editor
code mode must also receive ``pipeline_steps``.

.. code-block:: python

   from openhcs.constants.constants import AllComponents
   from openhcs.core.config import PipelineConfig
   from openhcs.core.source_bindings import (
       ComponentSelector,
       NamedSourceBinding,
       SourceBindingsConfig,
       SourceSelector,
   )

   pipeline_config = PipelineConfig(
       source_bindings_config=SourceBindingsConfig(
           bindings=(
               NamedSourceBinding(
                   alias="DNA",
                   selector=SourceSelector(
                       components=(
                           ComponentSelector(AllComponents.WELL, "A01"),
                           ComponentSelector(AllComponents.CHANNEL, "2"),
                       ),
                   ),
               ),
           ),
       ),
   )

Keep ``required=True`` (the default) when a missing source should stop
initialization. This makes misspelled filenames, absent channels, incompatible
datasets, and coordinate collisions visible before a pipeline runs.

Diagnostics and rich containers
-------------------------------

Plate inspection reports structured source diagnostics alongside the resolved
coordinates. Treat them as part of the dataset result, not as log decoration.
For Bio-Formats inputs, a packed-RGB series is reported with its exact OME image
and series identity and excluded from scalar microscopy planes. View or extract
that series with an RGB-capable tool; do not reinterpret its packed color bands
as independent OpenHCS channels.

If a container exposes only excluded/non-scalar series, or if its embedded
dataset or sample identity is ambiguous, initialization stops with a typed
error. Resolve the decoder or choose the exact dataset from authoritative
metadata instead of guessing a series number from filenames. Use bounded image
sampling to confirm a representative plane and its selected-resolution
provenance before authoring a full pipeline.
